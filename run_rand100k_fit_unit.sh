#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/fitter:${PYTHONPATH:-}"

N="${1:-100000}"

python3 scripts/angularfitter_masssweights.py \
  --settings plots/standard/data_qsq-1.1-7.0/results/0.yml \
  --data outputs/combined_with_sweights.root \
  --tree events \
  --toy \
  --nsig "$N" \
  --qsq 1.1 7.0
