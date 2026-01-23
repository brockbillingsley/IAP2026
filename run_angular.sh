#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate sweights-py312

export PYTHONPATH="$PWD/fitter:${PYTHONPATH:-}"

python -u scripts/angularfitter_masssweights.py --no-toy --polynomial standard --data outputs/combined_with_sweights.root --tree events --weight_branch w_sig --settings fitter/settings/App=0.1670_qsq-1.1-7.0.yml --qsq 1.1 7.0 --mKpi 0.64 1.60 2>&1 | tee run_angular.log
