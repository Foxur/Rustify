"""N1 training-throughput spike (docs/ROADMAP.md §6, gate C1).

    python -m arbitro_train.bench.train_throughput            # full plan, ~20-40 min
    python -m arbitro_train.bench.train_throughput --quick    # 3 cases, ~5 min

Measures full training steps (forward, proper-score loss, backward, grad clip, fused AdamW) of
encoder + 2-layer decision head with random weights and synthetic packed batches, so no data
or weights are needed. Modes:

  packed-compile  our packed ModernBERT, varlen FlashAttention-2, bf16 autocast, fp32 master
                  weights, torch.compile, static token budget (the planned trainer, §7.3)
  packed-eager    same without torch.compile
  padded-hf       transformers ModernBertModel (SDPA, padded to L) + the same head
  padded-hf-ckpt  same with gradient checkpointing (Laya's notebook recipe on a 4090)

Useful tokens/s counts real tokens only (no padding, no dummy tail). Gate C1: ModernBERT-large,
packed, L = 512 must reach >= 20k tokens/s; below that the GPU budget is re-planned (ADR-026).
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..config import PRESETS, EncoderConfig
from ..losses import gather_logits, proper_score_loss
from ..model import DecisionHead, DecisionModel, init_like_hf, varlen_available
from ..packing import PackSpec, synthetic_batch, to_device
from .common import host_info, md_table, write_report

PEAK_BF16_TFLOPS = 165.0  # RTX 4090 dense bf16 with fp32 accumulate (Ada whitepaper; ESTIMATED until gemm.py measures it)
C1_GATE = 20_000


@dataclass(frozen=True)
class Case:
    backbone: str
    mode: str
    seq_len: int
    budget: int = 12288
    freeze_embeddings: bool = False

    @property
    def name(self) -> str:
        return f"{self.backbone}/{self.mode}/L{self.seq_len}/T{self.budget}"


def plan(quick: bool, include_deberta: bool) -> list[Case]:
    if quick:
        return [
            Case("modernbert-large", "packed-compile", 512),
            Case("modernbert-large", "packed-eager", 512),
            Case("modernbert-large", "padded-hf-ckpt", 512),
        ]
    cases = [
        Case("modernbert-large", "packed-compile", 512),
        Case("modernbert-large", "packed-eager", 512),
        Case("modernbert-large", "packed-compile", 128),
        Case("modernbert-large", "packed-compile", 1024),
        Case("modernbert-large", "packed-compile", 512, budget=8192),
        Case("modernbert-large", "packed-compile", 512, budget=16384),
        Case("modernbert-large", "padded-hf", 512),
        Case("modernbert-large", "padded-hf-ckpt", 512),
        Case("modernbert-large", "padded-hf-ckpt", 1024),
        Case("modernbert-base", "packed-compile", 512),
        Case("modernbert-base", "packed-compile", 512, budget=24576),
        Case("mmbert-base", "packed-compile", 512, freeze_embeddings=True),
        Case("mmbert-base", "packed-compile", 1024, freeze_embeddings=True),
    ]
    if include_deberta:
        cases.append(Case("deberta-v3-large", "padded-hf", 512, budget=6144))
    return cases


def flops_per_token(cfg: EncoderConfig, avg_len: float, head_layers: int = 2) -> float:
    """Training FLOPs per real token (3x forward): dense GEMMs + attention scores/values."""
    d, n = cfg.hidden_size, cfg.num_hidden_layers
    dense = 2 * (n * (4 * d * d + 3 * d * cfg.intermediate_size))
    head = 2 * head_layers * (4 * d * d + 8 * d * d)
    n_global = sum(1 for t in cfg.layer_types if t == "full")
    keys_local = min(avg_len, 2 * cfg.window_one_side + 1)
    attn = 4 * d * (n_global * avg_len + (n - n_global) * keys_local) + 4 * d * head_layers * avg_len
    return 3.0 * (dense + head + attn)


def _hf_encoder(cfg: EncoderConfig, ckpt: bool):
    from transformers import ModernBertModel

    from ..selftest import hf_config

    m = ModernBertModel._from_config(hf_config(cfg), attn_implementation="sdpa")
    if ckpt:
        m.gradient_checkpointing_enable()
    return m


class PaddedModel(torch.nn.Module):
    """HF padded encoder + the same packed decision head (head cost identical across modes)."""

    def __init__(self, cfg: EncoderConfig, ckpt: bool):
        super().__init__()
        self.encoder = _hf_encoder(cfg, ckpt)
        self.head = DecisionHead(cfg.hidden_size, head_layers=2)

    def forward(self, b: dict, attn_impl="auto"):
        out = self.encoder(input_ids=b["pad_ids"], attention_mask=b["pad_mask"]).last_hidden_state
        h = out[b["pad_mask"].bool()]  # real tokens in packing order
        return self.head(h, b["token_qtype"][: h.shape[0]], b["cu_real"], b["max_seqlen"], b["marker_rows"], attn_impl)


def _padded_view(b: dict, seq_len: int, pad_id: int) -> dict:
    cu = b["cu_seqlens"]
    n = b["real_sequences"]
    lengths = np.diff(cu[: n + 1])
    ids = np.full((n, seq_len), pad_id, dtype=np.int64)
    mask = np.zeros((n, seq_len), dtype=np.int64)
    for i, (s, ln) in enumerate(zip(cu[:n], lengths)):
        ids[i, :ln] = b["input_ids"][s:s + ln]
        mask[i, :ln] = 1
    out = dict(b)
    out.update({"pad_ids": ids, "pad_mask": mask, "cu_real": cu[: n + 1].astype(np.int32)})
    return out


def _deberta_case(case: Case, steps: int, warmup: int) -> dict:
    from transformers import DebertaV2Config, DebertaV2Model

    cfg = DebertaV2Config(vocab_size=128100, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                          intermediate_size=4096, max_position_embeddings=512, relative_attention=True,
                          position_buckets=256, pos_att_type=["p2c", "c2p"], norm_rel_ebd="layer_norm",
                          share_att_key=True, max_relative_positions=-1, position_biased_input=False)
    model = DebertaV2Model(cfg).cuda()
    head = torch.nn.Linear(1024, 1).cuda()
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-5, fused=True)
    b_sz = case.budget // case.seq_len
    rng = np.random.default_rng(0)
    lengths = rng.integers(case.seq_len // 2, case.seq_len + 1, size=b_sz)
    ids = torch.randint(10, 128000, (b_sz, case.seq_len), device="cuda")
    mask = (torch.arange(case.seq_len, device="cuda")[None, :] < torch.tensor(lengths, device="cuda")[:, None]).long()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state
        loss = head(h[:, :8].float()).logsumexp(1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    return _time_steps(step, steps, warmup, int(lengths.sum()), int(b_sz * case.seq_len))


def _time_steps(step, steps: int, warmup: int, real_tokens_per_step: int, processed_per_step: int) -> dict:
    t0 = time.perf_counter()
    step()
    torch.cuda.synchronize()
    first = time.perf_counter() - t0
    for _ in range(max(0, warmup - 1)):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(steps):
        step()
    e.record()
    e.synchronize()
    sec = s.elapsed_time(e) / 1e3
    return {
        "first_step_s": round(first, 2),
        "step_ms": 1e3 * sec / steps,
        "tokens_per_s": real_tokens_per_step * steps / sec,
        "processed_tokens_per_s": processed_per_step * steps / sec,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def run_case(case: Case, steps: int, warmup: int, n_batches: int = 4) -> dict:
    if case.backbone == "deberta-v3-large":
        return _deberta_case(case, steps, warmup)
    cfg = PRESETS[case.backbone]
    spec = PackSpec(token_budget=case.budget, min_len=case.seq_len // 2, max_len=case.seq_len, max_options=20)
    host_batches = [synthetic_batch(spec, cfg.vocab_size, cfg.mask_token_id, seed=i, special_ids=(cfg.pad_token_id, cfg.mask_token_id))
                    for i in range(n_batches)]
    padded = case.mode.startswith("padded")
    if padded:
        host_batches = [_padded_view(b, case.seq_len, cfg.pad_token_id) for b in host_batches]
        model = PaddedModel(cfg, ckpt=case.mode.endswith("ckpt")).cuda()
    else:
        model = DecisionModel(cfg).cuda()
        init_like_hf(model.encoder)
    if case.freeze_embeddings:  # mmBERT: 196.6M embedding parameters stay frozen (docs/TRAINING.md §7.3)
        model.encoder.embeddings.tok_embeddings.weight.requires_grad_(False)
    batches = [to_device(b, "cuda") for b in host_batches]
    real = int(np.mean([b["real_tokens"] for b in host_batches]))
    processed = int(np.mean([b["pad_ids"].size if padded else case.budget for b in host_batches]))

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.dim() < 2 or "embeddings" in n or "type_emb" in n else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=1e-5, betas=(0.9, 0.98), eps=1e-6, fused=True)
    params = decay + no_decay
    fwd = torch.compile(model) if case.mode == "packed-compile" else model
    impl = "varlen"  # our head always runs packed; padded modes differ only in the HF encoder
    it = {"i": 0}

    def step():
        b = batches[it["i"] % len(batches)]
        it["i"] += 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = fwd(b, attn_impl=impl)
        logits = gather_logits(z, b["marker_index"], b["marker_mask"])
        loss, _ = proper_score_loss(logits, b["targets"], b["marker_mask"], b["qtype"], b["loss_weight"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    res = _time_steps(step, steps, warmup, real, processed)
    avg_len = float(np.mean([np.mean(np.diff(b["cu_seqlens"][: b["real_sequences"] + 1])) for b in host_batches]))
    res["mfu_vs_165"] = res["tokens_per_s"] * flops_per_token(cfg, avg_len) / (PEAK_BF16_TFLOPS * 1e12)
    res["avg_seq_len"] = avg_len
    res["real_tokens_per_step"] = real
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--deberta", action="store_true", help="add the DeBERTa-v3-large padded reference arm")
    ap.add_argument("--only", help="substring filter on case names, e.g. 'modernbert-large/packed'")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out-dir")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available() or not varlen_available():
        print("needs CUDA and torch.nn.attention.varlen (run python -m arbitro_train.bench.env first)")
        return 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rows = []
    for case in plan(args.quick, args.deberta):
        if args.only and args.only not in case.name:
            continue
        print(f"== {case.name}", flush=True)
        row = {"case": case.name, "backbone": case.backbone, "mode": case.mode, "L": case.seq_len, "T": case.budget}
        try:
            row.update(run_case(case, args.steps, args.warmup))
        except torch.OutOfMemoryError:
            row["error"] = "OOM"
        except Exception as e:  # keep going: one failing arm must not lose the night
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print(row, flush=True)
        rows.append(row)
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()

    c1 = [r for r in rows if r["backbone"] == "modernbert-large" and r["mode"].startswith("packed") and r["L"] == 512 and "tokens_per_s" in r]
    best = max((r["tokens_per_s"] for r in c1), default=float("nan"))
    verdict = "PASS" if best >= C1_GATE else ("FAIL" if not math.isnan(best) else "NOT MEASURED")
    print(f"C1 (ModernBERT-large packed L512): best {best:,.0f} tok/s -> {verdict} (gate {C1_GATE:,})", flush=True)

    cols = ["case", "tokens_per_s", "processed_tokens_per_s", "step_ms", "mfu_vs_165", "peak_alloc_gib", "peak_reserved_gib", "first_step_s", "error"]
    md = (f"# Training-throughput spike (N1)\n\nGate C1: ModernBERT-large, packed, L = 512 >= {C1_GATE:,} useful tokens/s. "
          f"Best measured: **{best:,.0f}** -> **{verdict}**.\n\n"
          "Useful tokens exclude padding and the dummy tail. MFU is against an assumed 165 TFLOP/s bf16 peak.\n\n")
    md += md_table([{**r, "mfu_vs_165": round(r["mfu_vs_165"], 3) if "mfu_vs_165" in r else ""} for r in rows], cols)
    write_report("train-throughput", {"host": host_info(), "gate_c1": {"best": best, "verdict": verdict}, "rows": rows,
                                      "steps": args.steps, "warmup": args.warmup}, md, args.out_dir)
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
