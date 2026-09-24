# Arbitro training (PyTorch)

The own-model track (`dm2`) trains with PyTorch 2.14; Rust owns data, calibration and evaluation
(ADR-023, [docs/TRAINING.md](../docs/TRAINING.md)). This directory currently holds the **M0 tooling**:
the packed ModernBERT trainer module and the GPU spikes that replace the plan's estimates with
measurements on the RTX 4090 ([ROADMAP §6](../docs/ROADMAP.md#6-the-first-two-weeks-task-by-task)).

| Path | What |
|---|---|
| `arbitro_train/model.py` | Packed (unpadded) ModernBERT on `torch.nn.attention.varlen.varlen_attn` (FlashAttention-2, window 64/64 on local layers), laya-v1-style decision head, HF weight loader |
| `arbitro_train/packing.py` | Static-shape packed micro-batches (token budget, dummy tail segment, zero-length padding segments) |
| `arbitro_train/losses.py` | Analytic proper-score loss: soft CE − 0.5·spherical + RPS (score questions) |
| `arbitro_train/selftest.py` | Correctness gate: packed module vs Hugging Face `ModernBertModel`, varlen vs dense, `torch.compile`, real weights |
| `arbitro_train/bench/env.py` | Bring-up check (driver, CUDA, bf16, varlen FA2 fwd+bwd, compile, VRAM, display, disk) |
| `arbitro_train/bench/gemm.py` | W1.3: GEMM TFLOP/s (bf16, fp16/fp32-acc, bf16→fp32 out, FP8, INT8) and windowed varlen FA2 |
| `arbitro_train/bench/train_throughput.py` | N1: full training steps, packed vs padded, gate C1 (≥ 20k useful tok/s, ModernBERT-large, L = 512) |
| `m0_gpu.sh` | Runs all of the above plus the Laya reference (`tools/goldens/`) with one log |
| `tests/` | CPU tests (`uv run pytest`) |

Verified in the design environment (CPU, torch 2.14.0, transformers 5.17.0): the packed encoder matches
`ModernBertModel` to 2.4e-7 on packed sequences longer than the sliding window, an off-by-one window
or a wrong RoPE θ makes that check fail, the padded benchmark arm computes the same function as the
packed arm, and the parameter counts equal Laya's encoders (394,781,696 / 306,939,648).
**Not yet run on a GPU:** the FlashAttention path, `torch.compile` and all numbers; the self-test is
the first thing to run on the 4090.

## Tonight on the RTX 4090

Requirements: **Linux** (native or WSL2; the varlen FlashAttention kernels and `torch.compile` need it),
NVIDIA driver ≥ 580 (the PyPI torch 2.14 wheels are CUDA 13), [`uv`](https://docs.astral.sh/uv/),
~20 GB free disk (Python environments ≈ 12 GB, Laya checkpoints 2.3 GB, ModernBERT-base 0.6 GB),
Hugging Face reachable. No account or token is needed for these public repositories.

```sh
git clone https://github.com/Foxur/Rustify && cd Rustify
git checkout claude/jev-layla-rust-2fs3dh
./training/m0_gpu.sh          # ~1–1.5 h, mostly unattended; stops early if a check fails
```

Or step by step (each writes `reports/spikes/<kind>-<host>-<stamp>.{json,md}`):

| # | Command (from `training/`) | Time | Decides |
|---|---|---|---|
| 1 | `uv sync` | 2–5 min | – |
| 2 | `uv run python -m arbitro_train.bench.env` | 1 min | must print `READY` |
| 3 | `uv run python -m arbitro_train.selftest --real modernbert-base` | 2–4 min | must print `SELFTEST PASSED`; otherwise no numbers are trusted |
| 4 | `uv run python -m arbitro_train.bench.gemm` | ~5 min | real bf16/FP8 ceilings (replaces the 165/330 TFLOP/s assumptions), cuBLAS bf16→fp32 output (ARCHITECTURE O-1) |
| 5 | `uv run python -m arbitro_train.bench.train_throughput` | 20–40 min | **C1**, micro-batch size (C2/C3), packed vs padded speed-up, compile gain |
| 6 | `cd ../tools/goldens && uv sync && uv run python fetch_checkpoints.py` | download 2.3 GB | on-disk dtype (CR G1), EN revision identity (CR §3 #9), sha256 pins |
| 7 | `uv run python laya_baseline.py --checkpoint laya-en --headroom` (repeat for `laya-multilingual`, `laya-typed-decisions`) | ~5 min each | P6 reference p50, saturated q/s, fp16 headroom (ADR-009), bf16 vs fp32 Δp |

Leave the GPU at its default power limit for these runs and do not use the desktop meanwhile; the
power limit is recorded in the env report. Afterwards commit the reports (they contain no weights):

```sh
git add reports/spikes tools/goldens/pins.lock.json
git commit -s -m "M0: spike reports from $(hostname)"
git push
```

### If something fails

- `varlen_attn_fwd_bwd` FAIL: not Linux, or torch older than 2.10; run inside WSL2 or native Linux.
- `driver_version` FAIL: update the NVIDIA driver to ≥ 580 (CUDA 13).
- `torch_compile` WARN: the `packed-compile` arms will fail; run `train_throughput --only packed-eager` and send the error.
- OOM on `T16384` or `padded-hf` arms is a result, not a bug: it is recorded in the table.
- Self-test FAIL: stop and send `reports/spikes/selftest-*.json` and the log; the throughput numbers would be meaningless.

## What comes next (E1)

Once C1 is measured, the first real training run is **E1, the early backbone signal**
([TRAINING.md §5.3](../docs/TRAINING.md#53-e1-the-early-signal-weeks-511)): ModernBERT-large vs
ModernBERT-base vs DeBERTa-v3-large on a licence-clean, gold-only mini-mixture (≈ 100–200k decisions
from CLINC150, PAWS, HellaSwag and GoEmotions, each with a manifest), ≈ 10–15 GPU-h. It answers the
largest model risk first: whether ModernBERT-large learns from a cold start (R2).
