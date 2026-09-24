"""W1.4: the PyTorch Laya reference on this machine - "the number to beat" (ADR-028).

    uv run python laya_baseline.py --checkpoint laya-en              # latency + throughput
    uv run python laya_baseline.py --checkpoint laya-en --headroom   # + fp16/bf16 max-abs sweep

Runs laya 0.3.7 (NandhaKishorM/laya @ 010bacef) unmodified, in-process, on a pinned checkpoint
fetched by fetch_checkpoints.py. Because laya rewrites tokenizer_config.json on load, it works on
a copy (the weights file is symlinked, not copied).

Measures (warm, in-process, end-to-end `Agent.system_one` incl. tokenization and post-processing,
and separately the model forward with CUDA events):
  * 1 question with a ~250-token sequence (gate P6 compares against this p50)
  * 5 and 50 questions per request over one state (P4 context)
  * saturated questions/s at 32/64/128 questions per request
--headroom additionally records max |activation| per Linear/LayerNorm in fp32 over a diverse input
set (fp16 range is +-65504), and the bf16-autocast vs fp32 probability difference (T4 context).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FP16_MAX = 65504.0

TICKET = ("Hello team, I ordered a pair of running shoes last week and was charged twice on my credit card. "
          "The second charge appeared yesterday and the order status still says processing. I contacted "
          "support via chat but nobody answered. Please refund the duplicate charge and tell me when the "
          "shoes will ship, because I need them for a race next weekend. Order number 48213, customer since 2019. ")

HEADROOM_STATES = [
    TICKET,
    TICKET * 6,
    "Sehr geehrte Damen und Herren, meine Rechnung vom 3. März ist doppelt abgebucht worden. Bitte erstatten Sie den Betrag.",
    "我的订单还没有发货，但是已经扣款两次了，请尽快退款。",
    {"ticket": {"id": 48213, "channel": "email", "body": TICKET, "priority": None, "tags": ["billing", "refund"]}},
    [{"role": "user", "content": "Can I cancel my subscription?"}, {"role": "assistant", "content": "Yes, in settings."},
     {"role": "user", "content": "It does not work, the button is greyed out!!!"}],
    "def charge(card, amount):\n    return gateway.capture(card, amount * 100)  # TODO: idempotency key\n" * 4,
    "| month | revenue | churn |\n|---|---|---|\n| Jan | 1,204,331.20 | 0.031 |\n| Feb | 998,120.00 | 0.044 |\n" * 3,
    "ok",
    "URGENT URGENT URGENT " * 40,
    "🙂👍🔥 great product but shipping 🐢 https://example.com/orders/48213?ref=mail#top",
    "The weather is nice today.",
]


def arbitro_home() -> Path:
    return Path(os.environ.get("ARBITRO_HOME", Path.home() / ".cache" / "arbitro"))


def resolve_checkpoint(cid: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    import tomllib

    e = tomllib.loads((HERE / "registry.toml").read_text())[cid]
    d = arbitro_home() / "checkpoints" / cid / e["revision"][:8]
    return d / e["subfolder"] if e.get("subfolder") else d


def working_copy(src: Path, cid: str) -> Path:
    """laya mutates tokenizer/tokenizer_config.json on load: give it a copy, symlink the weights."""
    dst = arbitro_home() / "work" / f"laya-baseline-{cid}"
    if dst.exists():
        shutil.rmtree(dst)
    for p in src.rglob("*"):
        if ".cache" in p.parts or not p.is_file():
            continue
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        if p.name == "model.safetensors":
            q.symlink_to(p.resolve())
        else:
            shutil.copy2(p, q)
    return dst


def choice_q(i: int) -> dict:
    return {"type": "choice", "instructions": f"Question {i}: which team should handle this ticket?",
            "criteria": {"billing": "payments, refunds, invoices", "shipping": "delivery and tracking",
                         "technical": "app or website problems", "other": None}}


def fit_state(agent, target_tokens: int) -> tuple[str, int]:
    """A ticket state whose single-question sequence is ~target_tokens long."""
    from laya.common import build_sequence

    words = (TICKET * 20).split()
    max_len = agent.cfg.get("max_len", 512)
    hml = agent.cfg.get("head_max_len", 192)
    q = agent._to_internal(choice_q(0))
    lo, hi = 1, len(words)
    while lo < hi:
        mid = (lo + hi) // 2
        n = len(build_sequence(agent.tok, " ".join(words[:mid]), q, max_len, hml)[0])
        if n < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    state = " ".join(words[:lo])
    return state, len(build_sequence(agent.tok, state, q, max_len, hml)[0])


class ForwardTimer:
    """CUDA-event time of every agent.model forward (pre/post hooks)."""

    def __init__(self, model):
        self.times: list[float] = []
        self._ev = None
        model.register_forward_pre_hook(self._pre)
        model.register_forward_hook(self._post)

    def _pre(self, *_):
        if torch.cuda.is_available():
            self._ev = torch.cuda.Event(enable_timing=True)
            self._ev.record()
        else:
            self._t = time.perf_counter()

    def _post(self, *_):
        if self._ev is not None:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            e.synchronize()
            self.times.append(self._ev.elapsed_time(e))
        else:
            self.times.append(1e3 * (time.perf_counter() - self._t))


def measure(agent, timer: ForwardTimer, state, n_questions: int, warmup: int, iters: int) -> dict:
    questions = {f"q{i}": choice_q(i) for i in range(n_questions)}
    for _ in range(warmup):
        agent.system_one(state, questions)
    timer.times.clear()
    e2e = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = agent.system_one(state, questions)
        e2e.append(1e3 * (time.perf_counter() - t0))
    fw = list(timer.times)
    e2e.sort()
    return {
        "questions": n_questions,
        "e2e_p50_ms": statistics.median(e2e),
        "e2e_p95_ms": e2e[min(len(e2e) - 1, int(0.95 * len(e2e)))],
        "forward_p50_ms": statistics.median(fw) if fw else None,
        "questions_per_s": n_questions / (statistics.median(e2e) / 1e3),
        "input_tokens": out["usage"]["input_tokens"],
    }


def headroom(agent) -> dict:
    """fp32 max-abs per Linear/LayerNorm output, and bf16-autocast vs fp32 probability deltas."""
    from laya.common import QTYPES, build_sequence, collate_items

    max_len = agent.cfg.get("max_len", 512)
    hml = agent.cfg.get("head_max_len", 192)
    qdefs = [choice_q(0),
             {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "somewhat", "urgent", "critical"]},
             {"type": "noul", "instructions": "The customer asks for a refund."}]
    items = []
    for st in HEADROOM_STATES:
        for qd in qdefs:
            q = agent._to_internal(qd)
            ids, markers = build_sequence(agent.tok, st, q, max_len, hml)
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
    maxabs: dict[str, float] = {}

    def register() -> list:
        hooks = []
        for name, mod in agent.model.named_modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.LayerNorm)):
                def hook(m, i, o, name=name):
                    o = o[0] if isinstance(o, tuple) else o
                    maxabs[name] = max(maxabs.get(name, 0.0), float(o.detach().abs().max()))
                hooks.append(mod.register_forward_hook(hook))
        return hooks

    dev = agent.device
    deltas = []
    for s in range(0, len(items), 8):
        b = collate_items([items[s:s + 8]], agent.tok.pad_token_id)
        args = [b[k].to(dev) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
        hooks = register()
        with torch.no_grad():  # no autocast: fp32 reference
            z32, _ = agent.model(*args)
        for h in hooks:
            h.remove()
        if dev.type == "cuda":
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                zbf, _ = agent.model(*args)
            m = b["marker_mask"].to(dev)
            p32 = torch.softmax(z32.float().masked_fill(~m, -1e4), -1)
            pbf = torch.softmax(zbf.float().masked_fill(~m, -1e4), -1)
            deltas.append(float((p32 - pbf).abs().max()))
    top = sorted(maxabs.items(), key=lambda kv: -kv[1])[:15]
    overall = top[0][1] if top else 0.0
    return {
        "sequences": len(items),
        "overall_max_abs": overall,
        "fp16_headroom_x": FP16_MAX / overall if overall else None,
        "top_modules": top,
        "bf16_vs_fp32_max_abs_dp": max(deltas) if deltas else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="laya-en")
    ap.add_argument("--dir", help="checkpoint directory (default: the pinned download in $ARBITRO_HOME)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--questions", default="1,5,50,32,64,128", help="questions per request to measure")
    ap.add_argument("--target-tokens", type=int, default=250)
    ap.add_argument("--headroom", action="store_true")
    ap.add_argument("--report-dir", default=str(REPO / "reports" / "spikes"))
    args = ap.parse_args(argv)

    import laya

    src = resolve_checkpoint(args.checkpoint, args.dir)
    if not (src / "model.safetensors").exists():
        print(f"checkpoint not found at {src}; run fetch_checkpoints.py first")
        return 1
    work = working_copy(src, args.checkpoint)
    agent = laya.load(str(work), device=args.device)
    timer = ForwardTimer(agent.model)
    state, seq_tokens = fit_state(agent, args.target_tokens)
    print(f"device={agent.device} dtype={agent.dtype} state sequence tokens={seq_tokens}", flush=True)

    rows = []
    for n in [int(x) for x in args.questions.split(",") if x]:
        r = measure(agent, timer, state, n, args.warmup, args.iters)
        rows.append(r)
        print(r, flush=True)
    result = {
        "checkpoint": args.checkpoint,
        "laya_version": getattr(laya, "__version__", "?"),
        "device": str(agent.device),
        "autocast_dtype": str(agent.dtype),
        "sequence_tokens_1q": seq_tokens,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
    }
    if args.headroom:
        result["headroom"] = headroom(agent)
        print(json.dumps(result["headroom"], indent=1), flush=True)

    rep = Path(args.report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    base = rep / f"laya-baseline-{args.checkpoint}-{socket.gethostname()}-{datetime.now():%Y%m%d-%H%M%S}"
    base.with_suffix(".json").write_text(json.dumps(result, indent=2))
    md = [f"# Laya PyTorch baseline: `{args.checkpoint}`\n",
          f"laya {result['laya_version']}, torch {torch.__version__}, {result['gpu'] or 'CPU'}, autocast {agent.dtype}, "
          f"1-question sequence = {seq_tokens} tokens, warm, in-process, {args.iters} iterations.\n",
          "| questions/request | e2e p50 ms | e2e p95 ms | forward p50 ms | questions/s | input_tokens |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        fw = f"{r['forward_p50_ms']:.2f}" if r["forward_p50_ms"] is not None else "-"
        md.append(f"| {r['questions']} | {r['e2e_p50_ms']:.2f} | {r['e2e_p95_ms']:.2f} | {fw} | {r['questions_per_s']:.1f} | {r['input_tokens']} |")
    if args.headroom:
        h = result["headroom"]
        md.append(f"\n## Activation headroom (fp32, {h['sequences']} sequences)\n")
        md.append(f"Overall max |activation| = {h['overall_max_abs']:.1f} -> fp16 headroom {h['fp16_headroom_x'] or 0:.2f}x. "
                  f"bf16-autocast vs fp32 max |dp| = {h['bf16_vs_fp32_max_abs_dp']}.\n")
        md.append("| module | max abs |\n|---|---|")
        md += [f"| `{n}` | {v:.1f} |" for n, v in h["top_modules"]]
    base.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"wrote {base}.json\nwrote {base}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
