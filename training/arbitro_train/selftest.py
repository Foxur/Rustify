"""Correctness checks for the packed trainer. Run before any benchmark or training run.

    python -m arbitro_train.selftest                     # CPU + (if present) CUDA checks, random weights
    python -m arbitro_train.selftest --real modernbert-base   # also compare real HF weights (downloads)

Checks
  1. [cpu]  PackedModernBert (sdpa path) == HF ModernBertModel, fp32, each sequence run alone,
            with packed sequences longer than the sliding window (window semantics, RoPE thetas,
            layer-0 norm, GeGLU, final norm). Tolerance 1e-5 (observed ~2e-7; an off-by-one
            window or a wrong RoPE theta gives >= 1e-4).
  2. [cuda] varlen (FlashAttention-2) path vs sdpa path under bf16 autocast, including zero-length
            padding segments and a dummy tail (static-shape packing). Forward and gradients.
  3. [cuda] torch.compile(forward+backward) of the full DecisionModel vs eager.
  4. [real] real checkpoint: ours vs HF in fp32, and ours bf16/varlen vs fp32.
Exit code 0 only if every executed check passes. Writes a JSON report with --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from .config import HF_REPOS, PRESETS, EncoderConfig, tiny_config
from .losses import gather_logits, proper_score_loss
from .model import DecisionModel, PackedModernBert, init_like_hf, load_encoder_state_dict, varlen_available
from .packing import PackSpec, batch_from_sequences, synthetic_batch, to_device

RESULTS: list[dict] = []


def record(name: str, ok: bool, **info) -> bool:
    RESULTS.append({"check": name, "ok": bool(ok), **info})
    flag = "PASS" if ok else "FAIL"
    detail = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items())
    print(f"[{flag}] {name}: {detail}", flush=True)
    return ok


def hf_config(cfg: EncoderConfig):
    from transformers import ModernBertConfig

    types = ["full_attention" if t == "full" else "sliding_attention" for t in cfg.layer_types]
    special = min(cfg.vocab_size - 1, 3)
    return ModernBertConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads, intermediate_size=cfg.intermediate_size,
        local_attention=cfg.local_attention, layer_types=types, norm_eps=cfg.norm_eps,
        rope_parameters={
            "full_attention": {"rope_theta": cfg.global_rope_theta, "rope_type": "default"},
            "sliding_attention": {"rope_theta": cfg.local_rope_theta, "rope_type": "default"},
        },
        pad_token_id=cfg.pad_token_id, bos_token_id=special, eos_token_id=special,
        cls_token_id=special, sep_token_id=special,
    )


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))


# ------------------------------------------------------------------------------------------ 1


def check_hf_parity(device: str, seed: int = 0) -> bool:
    from transformers import ModernBertModel

    torch.manual_seed(seed)
    cfg = tiny_config(local_attention=16, layers=4)  # window 8 per side
    hf = ModernBertModel._from_config(hf_config(cfg), attn_implementation="sdpa").to(device).eval()
    ours = PackedModernBert(cfg).to(device).eval()
    load_encoder_state_dict(ours, hf.state_dict())

    rng = np.random.default_rng(seed)
    lengths = [5, 37, 23, 64, 300]
    seqs = [rng.integers(4, cfg.vocab_size, size=n).tolist() for n in lengths]
    with torch.no_grad():
        ref = [hf(input_ids=torch.tensor([s], device=device)).last_hidden_state[0] for s in seqs]
        b = to_device(batch_from_sequences(seqs, [[0]] * len(seqs), [0] * len(seqs)), device)
        out = ours(b["input_ids"], b["position_ids"], b["cu_seqlens"], b["max_seqlen"], attn_impl="sdpa")
    ref_cat = torch.cat(ref)
    err = float((out - ref_cat).abs().max())
    ok = record("hf_parity_fp32", err < 1e-5, device=device, max_abs_err=err, lengths=str(lengths))

    # Same with a dummy tail and zero-length padding segments (static-shape packing).
    b2 = to_device(batch_from_sequences(seqs, [[0]] * len(seqs), [0] * len(seqs),
                                        token_budget=sum(lengths) + 11, max_segments=len(seqs) + 3), device)
    with torch.no_grad():
        out2 = ours(b2["input_ids"], b2["position_ids"], b2["cu_seqlens"], b2["max_seqlen"], attn_impl="sdpa")
    err2 = float((out2[: sum(lengths)] - ref_cat).abs().max())
    ok &= record("hf_parity_fp32_padded_stream", err2 < 1e-5, device=device, max_abs_err=err2)
    return ok


# ------------------------------------------------------------------------------------------ 2


def _medium_cfg() -> EncoderConfig:
    base = PRESETS["modernbert-base"]
    return EncoderConfig(**{**base.to_dict(), "num_hidden_layers": 6,
                            "layer_types": tuple("full" if i % 3 == 0 else "sliding" for i in range(6)),
                            "vocab_size": 4096, "pad_token_id": 0, "mask_token_id": 1, "name": "medium"})


def _batch(cfg: EncoderConfig, budget: int, seed: int, device: str) -> dict:
    spec = PackSpec(token_budget=budget, min_len=96, max_len=320, max_options=12)
    return to_device(synthetic_batch(spec, cfg.vocab_size, cfg.mask_token_id, seed, special_ids=(0, 1)), device)


def _step(model: DecisionModel, b: dict, impl: str, amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        z = model(b, attn_impl=impl)
    logits = gather_logits(z, b["marker_index"], b["marker_mask"])
    loss, _ = proper_score_loss(logits, b["targets"], b["marker_mask"], b["qtype"], b["loss_weight"])
    return z, loss


def check_varlen_vs_sdpa(seed: int = 0) -> bool:
    torch.manual_seed(seed)
    cfg = _medium_cfg()
    model = DecisionModel(cfg, head_layers=2, dropout=0.0).cuda()
    init_like_hf(model.encoder)
    b = _batch(cfg, 2048, seed, "cuda")
    real_markers = b["real_markers"]

    grads = {}
    outs = {}
    for impl in ("sdpa", "varlen"):
        model.zero_grad(set_to_none=True)
        z, loss = _step(model, b, impl, amp=True)
        loss.backward()
        outs[impl] = z[:real_markers].detach()
        grads[impl] = torch.cat([p.grad.flatten().float() for p in model.parameters() if p.grad is not None])
    fc = cosine(outs["varlen"], outs["sdpa"])
    c = cosine(grads["varlen"], grads["sdpa"])
    ok = record("varlen_vs_sdpa_bf16_forward", fc > 0.999, cosine=fc, rel_err=rel_err(outs["varlen"], outs["sdpa"]))
    ok &= record("varlen_vs_sdpa_bf16_grad", c > 0.99, grad_cosine=c)

    # Window semantics on the FA2 path: compare against an fp32 sdpa reference on a real window.
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        z32, _ = _step(model.float(), b, "sdpa", amp=False)
        zbf, _ = _step(model, b, "varlen", amp=True)
    c32 = cosine(zbf[:real_markers], z32[:real_markers])
    ok &= record("varlen_bf16_vs_sdpa_fp32", c32 > 0.995, cosine=c32, rel_err=rel_err(zbf[:real_markers], z32[:real_markers]))
    return ok


# ------------------------------------------------------------------------------------------ 3


def check_compile(seed: int = 0) -> bool:
    torch.manual_seed(seed)
    cfg = _medium_cfg()
    model = DecisionModel(cfg, head_layers=2, dropout=0.0).cuda()
    init_like_hf(model.encoder)
    compiled = torch.compile(model)
    ok = True
    for i in range(3):  # same static shape each step: must not recompile
        b = _batch(cfg, 2048, seed + i, "cuda")
        t0 = time.perf_counter()
        z_c, loss_c = _step(compiled, b, "varlen", amp=True)
        loss_c.backward()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        with torch.no_grad():
            z_e, _ = _step(model, b, "varlen", amp=True)
        zc, ze = z_c[: b["real_markers"]].detach(), z_e[: b["real_markers"]]
        c = cosine(zc, ze)
        ok &= record(f"compile_step{i}", c > 0.999, cosine=c, rel_err=rel_err(zc, ze), seconds=dt)
        model.zero_grad(set_to_none=True)
    return ok


# ------------------------------------------------------------------------------------------ 4


def check_real(name: str, device: str) -> bool:
    from transformers import AutoTokenizer, ModernBertModel

    repo = HF_REPOS.get(name, name)
    hf = ModernBertModel.from_pretrained(repo, attn_implementation="sdpa", dtype=torch.float32).to(device).eval()
    cfg = EncoderConfig.from_hf_dict(hf.config.to_dict(), name=name)
    ours = PackedModernBert(cfg).to(device).eval()
    load_encoder_state_dict(ours, hf.state_dict())
    tok = AutoTokenizer.from_pretrained(repo)
    texts = [
        "The customer was charged twice for the same order and asks for a refund.",
        " ".join(["Long context sentence number %d about billing and shipping." % i for i in range(40)]),
    ]
    seqs = [tok(t)["input_ids"] for t in texts]
    with torch.no_grad():
        ref = torch.cat([hf(input_ids=torch.tensor([s], device=device)).last_hidden_state[0] for s in seqs])
        b = to_device(batch_from_sequences(seqs, [[0]] * 2, [0] * 2), device)
        out = ours(b["input_ids"], b["position_ids"], b["cu_seqlens"], b["max_seqlen"], attn_impl="sdpa")
    err = float((out - ref).abs().max())
    ok = record(f"real_{name}_fp32", err < 2e-3, max_abs_err=err, tokens=int(b["real_tokens"]))
    if device == "cuda" and varlen_available():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outb = ours(b["input_ids"], b["position_ids"], b["cu_seqlens"], b["max_seqlen"], attn_impl="varlen")
        c = cosine(outb, ref)
        ok &= record(f"real_{name}_bf16_varlen_vs_fp32", c > 0.999, cosine=c)
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="append", default=[], help="also check real weights: modernbert-base | modernbert-large | mmbert-base | <hf repo>")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--out", help="write a JSON report here")
    args = ap.parse_args(argv)

    cuda = torch.cuda.is_available() and not args.cpu_only
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if cuda else "cpu",
            "varlen_available": varlen_available()}
    print(json.dumps(info), flush=True)
    ok = check_hf_parity("cpu")
    if cuda:
        ok &= check_hf_parity("cuda")
        if varlen_available():
            ok &= check_varlen_vs_sdpa()
            if not args.skip_compile:
                ok &= check_compile()
        else:
            ok &= record("varlen_available", False, hint="torch.nn.attention.varlen.varlen_attn missing: need torch>=2.10 on Linux with CUDA")
    for name in args.real:
        ok &= check_real(name, "cuda" if cuda else "cpu")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"info": info, "results": RESULTS, "ok": bool(ok)}, f, indent=2)
    print("SELFTEST", "PASSED" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
