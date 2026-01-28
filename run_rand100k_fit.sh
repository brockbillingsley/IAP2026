#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sweights-py312

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/fitter:$REPO_DIR:${PYTHONPATH:-}"

python scripts/angularfitter_masssweights.py \
  --data outputs/combined_with_sweights.root \
  --tree events \
  --settings plots/standard/data_qsq-1.1-7.0/results/0.yml \
  --polynomial standard \
  --qsq 1.1 7.0 \
  --mKpi 0.796 0.996 \
  --toy \
  --nsig 100000
