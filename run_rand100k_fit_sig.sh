#!/usr/bin/env bash
set -euo pipefail

# Optional: first arg overrides sample size (default 100000)
N="${1:-100000}"

python3 scripts/angularfitter_masssweights.py \
  --settings plots/standard/data_qsq-1.1-7.0/results/0.yml \
  --data outputs/combined_with_sweights.root \
  --tree events \
  --weight_branch w_sig \
  --toy \
  --nsig "$N" \
  --qsq 1.1 7.0
