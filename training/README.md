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
| `arbitro_train/data/` | E1 sources (fetch + convert, manifests in `data/manifests/`) and the question/sequence/packing pipeline |
| `arbitro_train/train.py`, `configs/e1-*.toml`, `e1.sh` | The E1 spike trainer and its runner; `report_e1.py` summarises runs |
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

**Windows machine → WSL2 with Ubuntu 24.04** (native Windows is not supported: no varlen FlashAttention
and no Triton for `torch.compile`):

```powershell
wsl --install -d Ubuntu-24.04     # PowerShell as administrator; then: wsl --update
```

Install the NVIDIA driver (≥ 580) on **Windows only**; inside WSL there is no Linux driver and no CUDA
toolkit (the torch wheels bring the CUDA runtime). Give WSL enough memory in `%UserProfile%\.wslconfig`
(`[wsl2]`, `memory=48GB`, `swap=16GB`, then `wsl --shutdown`). The GPU power limit can only be changed
with the Windows `nvidia-smi`. Keep the repository and data inside the Linux file system (`~/`), never
under `/mnt/c` (much slower I/O).

```sh
sudo apt update && sudo apt install -y build-essential git curl   # Triton needs gcc
curl -LsSf https://astral.sh/uv/install.sh | sh && exec $SHELL
nvidia-smi                                                       # must show the RTX 4090
git clone https://github.com/Foxur/Rustify && cd Rustify
git checkout claude/jev-layla-rust-2fs3dh
./training/m0_gpu.sh          # ~1–1.5 h, mostly unattended; stops early if a check fails
./training/e1.sh              # then the first real training run (below)
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

## E1: the first real training run

After `m0_gpu.sh` passed its self-test, run the early backbone signal
([TRAINING.md §5.3](../docs/TRAINING.md#53-e1-the-early-signal-weeks-511)):

```sh
./training/e1.sh            # data, then ModernBERT-base (~15-20 min) and ModernBERT-large (~35-45 min)
./training/e1.sh large      # one arm only
```

| Step | What happens |
|---|---|
| Data | `python -m arbitro_train.data.sources` downloads the sources named in `data/manifests/*.toml` (pinned commits, sha256-checked; PAWS and ANLI are recorded trust-on-first-use in `data/data.lock.json`) and converts them to `$ARBITRO_HOME/data/e1/` (default `~/.cache/arbitro`) |
| Training mixture (gold only, no teachers) | CLINC150 (intent choice over 5-20 sampled intents, out-of-scope → "none of the listed options"), GoEmotions (emotion choice or noul), PAWS (paraphrase noul, 30 % negated with flipped label); mixed by p ∝ n^0.4 |
| Evaluation | In-domain dev (CLINC150, GoEmotions, PAWS) and OOD-S dev: MASSIVE-en dev (20 options, fixed seed 13) and ANLI dev (evaluation only, CC BY-NC) |
| Model | Pretrained encoder from the Hub (`answerdotai/ModernBERT-*`, Apache-2.0) + 2-layer decision head, layout L0 (Laya's sequence layout, token-identical to `laya.common.build_sequence`, tested), hybrid read-out |
| Optimisation | bf16 autocast, fp32 master weights, fused AdamW (0.9, 0.98), LR 2e-5 encoder / 3e-4 head, layer-wise decay 0.9, 6 % warmup + cosine, 50M tokens, analytic proper-score loss |
| Output | `runs/<stamp>-<config>-s<seed>/` (metrics.jsonl, last.pt for `--resume`, best/model.safetensors; git-ignored) and `reports/e1/e1-<host>-<stamp>.md` (commit this) |

A run can be continued after an interruption with
`uv run python -m arbitro_train.train --resume runs/<run>` (it restarts at the beginning of the current data epoch).
Overrides work without editing files, e.g. `--set train.token_budget=8192 --set train.total_tokens=20000000`.

The E1 numbers are **signals, not model results**: an arm "learns" when its OOD-S dev accuracy has a 95 % CI lower
bound above chance. The question E1 answers first is risk R2: does ModernBERT-large learn from a cold start at all?
The DeBERTa-v3-large reference arm and the layout-L2 arm follow once these two arms have run.

Licences: every source manifest is `status = "reported"` until you sign the checklist in it (`verified_by`,
`verified_on`); the trainer prints a warning for each. E1 is a signal run; nothing trained here is released.
HellaSwag is not in this first mixture: its GitHub source (`rowanz/hellaswag`) was not reachable on 2026-09-24.
