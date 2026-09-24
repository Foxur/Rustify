"""E1 spike trainer: packed ModernBERT + decision head on the gold-only mini-mixture.

    python -m arbitro_train.train --config configs/e1-modernbert-large.toml
    python -m arbitro_train.train --config configs/e1-modernbert-base.toml --set train.seed=2
    python -m arbitro_train.train --config ... --resume runs/<run>          # continue a run

Setup (docs/TRAINING.md §5.3, §7): layout L0, hybrid read-out, bf16 autocast with fp32 master
weights, fused AdamW (0.9, 0.98, eps 1e-6, wd 0.01 without norms/biases/embeddings), layer-wise LR
decay, 6 % linear warmup then cosine, grad clip 1.0, packed static-shape micro-batches with
gradient accumulation, optional torch.compile. The analytic proper-score loss (losses.py).

Every ``eval_every_tokens`` it evaluates the in-domain dev sets and the OOD-S dev sets (MASSIVE-en
dev, ANLI dev), appends to runs/<run>/metrics.jsonl, saves runs/<run>/last.pt and exports the best
OOD-S model to runs/<run>/best/. E1 numbers are signals, never model results (TRAINING.md §5.3):
an arm "learns" when its OOD-S accuracy CI lower bound is above chance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import socket
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from .config import tiny_config
from .data.pipeline import L0Builder, TrainStream, build_eval_batches, load_tokenizer
from .data.sources import load_records
from .losses import gather_logits, proper_score_loss
from .metrics import Accumulator
from .model import DecisionModel, init_like_hf, load_pretrained_encoder
from .packing import StaticShape, to_device

REPO = Path(__file__).resolve().parents[2]


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = tomllib.loads(Path(path).read_text())
    for ov in overrides:
        key, val = ov.split("=", 1)
        try:
            val = tomllib.loads(f"v = {val}")["v"]
        except tomllib.TOMLDecodeError:
            pass  # bare string
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


def param_groups(model: DecisionModel, t: dict) -> list[dict]:
    n = model.encoder.config.num_hidden_layers
    groups: dict[tuple[float, float], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            sub = name[len("encoder."):]
            if sub.startswith("embeddings."):
                depth = 0
            elif sub.startswith("layers."):
                depth = int(sub.split(".")[1]) + 1
            else:
                depth = n + 1
            lr = t["lr_encoder"] * t["llrd"] ** (n + 1 - depth)
        else:
            lr = t["lr_head"]
        no_decay = p.dim() < 2 or "embeddings" in name or "type_emb" in name or "norm" in name
        groups.setdefault((lr, 0.0 if no_decay else t["weight_decay"]), []).append(p)
    return [{"params": ps, "lr": lr, "base_lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]


def lr_factor(progress: float, warmup: float) -> float:
    if progress < warmup:
        return progress / max(warmup, 1e-9)
    x = min(1.0, (progress - warmup) / max(1e-9, 1.0 - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * x))


def split_spec(spec: str) -> tuple[str, str]:
    src, _, split = spec.partition(":")
    return src, split or "dev"


def evaluate(fwd, eval_sets: dict, device: torch.device, amp: bool) -> dict:
    acc = Accumulator()
    with torch.no_grad():
        for name, batches in eval_sets.items():
            for hb in batches:
                b = to_device(hb, device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    z = fwd(b)
                logits = gather_logits(z, b["marker_index"], b["marker_mask"]).float().cpu().numpy()
                mask = hb["marker_mask"]
                for s in range(hb["real_sequences"]):
                    k = int(mask[s].sum())
                    zz = logits[s, :k]
                    p = np.exp(zz - zz.max())
                    acc.add(name, p / p.sum(), int(hb["gold"][s]))
    return acc.summary()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="TOML config (not needed with --resume)")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.total_tokens=20000000")
    ap.add_argument("--run-dir")
    ap.add_argument("--resume", help="run directory to continue")
    ap.add_argument("--tiny", action="store_true", help="tiny random model on CPU (pipeline test)")
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args(argv)

    if args.resume:
        run_dir = Path(args.resume)
        cfg = json.loads((run_dir / "config.json").read_text())
    else:
        if not args.config:
            ap.error("--config is required unless --resume is given")
        cfg = load_config(args.config, args.set)
        name = Path(args.config).stem + ("-tiny" if args.tiny else "")
        run_dir = Path(args.run_dir or REPO / "runs" / f"{dt.datetime.now():%Y%m%d-%H%M%S}-{name}-s{cfg['train']['seed']}")
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg["meta"] = {"host": socket.gethostname(), "torch": torch.__version__, "tiny": args.tiny,
                       "started": dt.datetime.now().isoformat(timespec="seconds")}
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    m, d, t = cfg["model"], cfg["data"], cfg["train"]
    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.tiny else "cpu")
    amp = device.type == "cuda"
    print(f"run {run_dir} on {device}", flush=True)

    # ---------------------------------------------------------------- data
    train_sources = {}
    for s in d["train"]:
        recs, meta = load_records(s)
        if "train" not in meta["allowed_use"]:
            raise SystemExit(f"{s}: licence manifest does not allow training (allowed_use={meta['allowed_use']})")
        if meta["licence_status"] != "verified":
            print(f"WARNING {s}: licence status '{meta['licence_status']}' - sign the checklist in data/manifests/{s}.toml "
                  "before any release; E1 is a signal run only", flush=True)
        train_sources[s] = ([r for r in recs if r["split"] == "train"], meta)
    tok = load_tokenizer(m["tokenizer"])
    shape = StaticShape(token_budget=t["token_budget"], max_segments=t["max_segments"], k_max=t["k_max"],
                        max_seqlen=m["max_len"], span_max=t["token_budget"] // 2)
    builder = L0Builder(tok, m["max_len"], m["head_max_len"])
    eval_sets, eval_groups = {}, {"in": [], "ood": []}
    for group in ("eval_in", "eval_ood"):
        for spec in d.get(group, []):
            src, split = split_spec(spec)
            try:
                recs, meta = load_records(src)
            except FileNotFoundError as e:
                print(f"skip eval {spec}: {e}", flush=True)
                continue
            recs = [r for r in recs if r["split"] == split]
            batches, n_q = build_eval_batches(recs, meta, -1, builder, shape, limit=d.get("eval_limit"))
            eval_sets[spec] = batches
            eval_groups["in" if group == "eval_in" else "ood"].append(spec)
            print(f"eval {spec}: {n_q} questions in {len(batches)} micro-batches", flush=True)

    # ---------------------------------------------------------------- model
    if args.tiny:
        enc_cfg = tiny_config(layers=2, vocab=len(tok))
        enc_cfg = type(enc_cfg)(**{**enc_cfg.to_dict(), "pad_token_id": tok.pad_token_id, "mask_token_id": tok.mask_token_id})
        model = DecisionModel(enc_cfg, m["head_layers"], m["dropout"], m["readout"])
        init_like_hf(model.encoder)
    else:
        encoder = load_pretrained_encoder(m["hf_repo"], None, m.get("revision") or None)  # shapes from the checkpoint config
        model = DecisionModel(encoder.config, m["head_layers"], m["dropout"], m["readout"])
        model.encoder = encoder
    for tid in (tok.cls_token_id, tok.sep_token_id, tok.mask_token_id, tok.pad_token_id):
        if tid >= model.encoder.config.vocab_size:
            raise SystemExit(f"tokenizer id {tid} >= encoder vocab {model.encoder.config.vocab_size}: tokenizer and backbone differ")
    model.to(device)
    opt = torch.optim.AdamW(param_groups(model, t), betas=tuple(t["betas"]), eps=t["eps"], fused=(device.type == "cuda"))
    state = {"step": 0, "tokens": 0, "best": -1.0, "epoch": 0}
    if args.resume:
        ck = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        torch.set_rng_state(ck["torch_rng"])
        print(f"resumed at step {state['step']}, {state['tokens']:,} tokens, epoch {state['epoch']}", flush=True)
    fwd = torch.compile(model) if (t.get("compile") and device.type == "cuda") else model

    stream = TrainStream(train_sources, m["tokenizer"], shape, m["max_len"], m["head_max_len"],
                         seed=t["seed"] + 1000 * state["epoch"], tau=d.get("tau", 0.4))
    nw = t.get("num_workers", 4)
    loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=nw, pin_memory=(device.type == "cuda"),
                                         prefetch_factor=4 if nw else None, persistent_workers=bool(nw))
    it = iter(loader)
    log = open(run_dir / "metrics.jsonl", "a")

    def do_eval(final: bool = False):
        model.eval()
        t0 = time.perf_counter()
        res = evaluate(fwd, eval_sets, device, amp)
        model.train()
        ood = [res[k]["acc"] for k in eval_groups["ood"] if k in res]
        ood_macro = float(np.mean(ood)) if ood else float("nan")
        rec = {"kind": "eval", "step": state["step"], "tokens": state["tokens"], "ood_macro_acc": ood_macro,
               "sets": res, "seconds": round(time.perf_counter() - t0, 1), "final": final}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        line = " ".join(f"{k}={v['acc']:.3f}(ch {v['chance']:.2f},ece {v['ece15']:.3f})" for k, v in res.items())
        print(f"[eval] step {state['step']} tokens {state['tokens']:,} ood_macro={ood_macro:.4f} | {line}", flush=True)
        return ood_macro, res

    def save(best: bool):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state, "torch_rng": torch.get_rng_state()},
                   run_dir / "last.pt.tmp")
        os.replace(run_dir / "last.pt.tmp", run_dir / "last.pt")
        if best:
            from safetensors.torch import save_file

            bd = run_dir / "best"
            bd.mkdir(exist_ok=True)
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(bd / "model.safetensors"))
            (bd / "arbitro-e1.json").write_text(json.dumps({"config": cfg, "state": state, "encoder": model.encoder.config.to_dict(),
                                                             "tokenizer": m["tokenizer"], "layout": "L0"}, indent=2, default=list))

    total, accum = t["total_tokens"], t["grad_accum"]
    last_eval: dict = {"step": -1, "res": None}
    next_eval = (state["tokens"] // t["eval_every_tokens"] + 1) * t["eval_every_tokens"]
    model.train()
    t_last, tok_last = time.perf_counter(), state["tokens"]
    run_loss = run_acc = 0.0
    run_n = 0
    while state["tokens"] < total:
        for _ in range(accum):
            hb = next(it)
            state["epoch"] = int(hb.pop("epoch"))
            b = to_device(hb, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                z = fwd(b)
            logits = gather_logits(z, b["marker_index"], b["marker_mask"])
            loss, st = proper_score_loss(logits, b["targets"], b["marker_mask"], b["qtype"], b["loss_weight"],
                                         w_sph=t.get("w_sph", 0.5), w_rps=t.get("w_rps", 1.0))
            (loss / accum).backward()
            state["tokens"] += int(hb["real_tokens"])
            run_loss += loss.detach()
            run_acc += st["acc"]
            run_n += 1
        progress = state["tokens"] / total
        f = lr_factor(progress, t["warmup_frac"])
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        state["step"] += 1
        if state["step"] % t.get("log_every_steps", 10) == 0:
            now = time.perf_counter()
            tps = (state["tokens"] - tok_last) / max(now - t_last, 1e-9)
            rec = {"kind": "train", "step": state["step"], "tokens": state["tokens"], "epoch": state["epoch"],
                   "loss": float(run_loss / run_n), "acc": float(run_acc / run_n), "lr_factor": f,
                   "grad_norm": float(gnorm), "tokens_per_s": tps,
                   "mem_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {state['step']} tok {state['tokens']:,} ({100 * progress:.1f}%) ep {state['epoch']} "
                  f"loss {rec['loss']:.4f} acc {rec['acc']:.3f} lr×{f:.3f} gn {rec['grad_norm']:.2f} "
                  f"{tps:,.0f} tok/s mem {rec['mem_gib']:.1f} GiB", flush=True)
            t_last, tok_last = now, state["tokens"]
            run_loss = run_acc = 0.0
            run_n = 0
        if state["tokens"] >= next_eval or (args.max_steps and state["step"] >= args.max_steps):
            ood_macro, last_eval["res"] = do_eval()
            last_eval["step"] = state["step"]
            improved = not math.isnan(ood_macro) and ood_macro > state["best"]
            if improved:
                state["best"] = ood_macro
            save(best=improved)
            next_eval += t["eval_every_tokens"]
        if args.max_steps and state["step"] >= args.max_steps:
            break

    if last_eval["step"] == state["step"] and last_eval["res"] is not None:
        res = last_eval["res"]  # the loop just evaluated this step
    else:
        ood_macro, res = do_eval(final=True)
        if not math.isnan(ood_macro) and ood_macro > state["best"]:
            state["best"] = ood_macro
            save(best=True)
        else:
            save(best=False)
    verdict = {k: {"acc": res[k]["acc"], "ci95": res[k]["acc_ci95"], "chance": res[k]["chance"],
                   "learns": res[k]["acc_ci95"][0] > res[k]["chance"]} for k in eval_groups["ood"] if k in res}
    summary = {"run": str(run_dir), "tokens": state["tokens"], "steps": state["step"], "best_ood_macro": state["best"],
               "final": res, "e1_signal": verdict}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("E1 signal (OOD-S dev, CI lower bound above chance = learns):")
    for k, v in verdict.items():
        print(f"  {k}: acc {v['acc']:.3f} CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] chance {v['chance']:.3f} -> "
              f"{'LEARNS' if v['learns'] else 'no signal'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
