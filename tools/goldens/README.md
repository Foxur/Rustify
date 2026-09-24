# tools/goldens: the pinned Laya reference (ADR-012)

A separate uv project, because the reference environment is pinned tighter than the trainer:
CPython 3.11, laya 0.3.7 installed **from git at `010bacef`** (0.3.7 is no longer on PyPI, AM-17),
torch 2.14.0, transformers 5.17.0, tokenizers 0.23.2, numpy 2.4.6.

| File | What |
|---|---|
| `registry.toml` | The pinned checkpoints (ADR-005): repo, 40-hex revision, subfolder |
| `fetch_checkpoints.py` | W1.2: downloads them to `$ARBITRO_HOME/checkpoints` (default `~/.cache/arbitro`), records per-file sha256, safetensors dtypes and counts, config values → `pins.lock.json` + `reports/spikes/checkpoints-*.md` |
| `laya_baseline.py` | W1.4: unmodified laya on this machine: 1/5/50-question latency, saturated q/s, `--headroom` fp32 max-abs sweep and bf16-vs-fp32 Δp → `reports/spikes/laya-baseline-*` |
| `pins.lock.json` | Written by `fetch_checkpoints.py`; commit it. Contains hashes, never weights |

```sh
uv sync
uv run python fetch_checkpoints.py
uv run python laya_baseline.py --checkpoint laya-en --headroom
```

Laya weights are never committed, mirrored or put in images (ADR-030). laya rewrites
`tokenizer/tokenizer_config.json` on load, so `laya_baseline.py` runs on a copy and the pinned
directories keep their recorded sha256. Golden fixture generation (W2.4, L1–L3) lands here next.
