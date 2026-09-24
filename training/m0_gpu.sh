#!/usr/bin/env bash
# M0 GPU session on the training machine (docs/ROADMAP.md §6: W1.2, W1.3, W1.4, N1).
# Usage:  ./training/m0_gpu.sh            # everything, ~1-1.5 h, mostly unattended
#         ./training/m0_gpu.sh --no-laya  # skip the Laya checkpoint download and baseline
# Every step writes reports/spikes/*.{json,md}; the combined log goes to reports/spikes/m0-<host>-<stamp>.log.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$ROOT/reports/spikes/m0-$(hostname)-$STAMP.log"
mkdir -p "$ROOT/reports/spikes"
exec > >(tee -a "$LOG") 2>&1

LAYA=1
[[ "${1:-}" == "--no-laya" ]] && LAYA=0
step() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

step "1/6 training env (uv sync)"
(cd training && uv sync) || { echo "uv sync failed"; exit 1; }

step "2/6 environment check"
(cd training && uv run python -m arbitro_train.bench.env) || { echo "environment not ready - see FAIL lines above"; exit 1; }

step "3/6 self-test (packed ModernBERT vs Hugging Face, varlen FA2, torch.compile, real ModernBERT-base weights)"
(cd training && uv run python -m arbitro_train.selftest --real modernbert-base --out "$ROOT/reports/spikes/selftest-$(hostname)-$STAMP.json") \
  || { echo "self-test failed - do not trust the throughput numbers; send the log"; exit 1; }

step "4/6 CUDA microbenchmarks (GEMM bf16/fp16/fp8/int8, varlen FlashAttention)"
(cd training && uv run python -m arbitro_train.bench.gemm) || echo "gemm bench failed (continuing)"

step "5/6 training-throughput spike N1 (gate C1)"
(cd training && uv run python -m arbitro_train.bench.train_throughput)
echo "train_throughput exit code: $? (0 = C1 PASS, 2 = C1 FAIL, other = error)"

if [[ $LAYA == 1 ]]; then
  step "6/6 Laya reference: pinned checkpoints + PyTorch baseline"
  (cd tools/goldens && uv sync && uv run python fetch_checkpoints.py) || { echo "checkpoint fetch failed"; exit 1; }
  for c in laya-en laya-multilingual laya-typed-decisions; do
    (cd tools/goldens && uv run python laya_baseline.py --checkpoint "$c" --headroom) || echo "baseline $c failed (continuing)"
  done
fi

step "done - commit the results:"
echo "  git add reports/spikes tools/goldens/pins.lock.json && git commit -s -m 'M0: spike reports from $(hostname)' && git push"
