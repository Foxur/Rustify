#!/usr/bin/env bash
# E1 early backbone signal (docs/TRAINING.md §5.3) on the RTX 4090.
# Usage:  ./training/e1.sh              # ModernBERT-base, then ModernBERT-large (~1-1.5 h incl. downloads)
#         ./training/e1.sh large        # one arm only (base | large)
# Run ./training/m0_gpu.sh first: its self-test must pass and C1 decides the token budget.
set -euo pipefail
cd "$(dirname "$0")"
ARMS=${1:-"base large"}
[[ "$ARMS" == "both" ]] && ARMS="base large"

uv sync
echo "=== data: training sources are required, ANLI (evaluation only) is optional"
uv run python -m arbitro_train.data.sources --only clinc150 --only goemotions --only paws --only massive-en
uv run python -m arbitro_train.data.sources --only anli --skip-failed || true

for arm in $ARMS; do
  echo "=== E1 arm: modernbert-$arm"
  uv run python -m arbitro_train.train --config "configs/e1-modernbert-$arm.toml"
done
uv run python -m arbitro_train.report_e1
echo "Commit the report (no weights): git add reports/e1 data/data.lock.json && git commit -s -m 'E1: first signal' && git push"
