#!/bin/bash
#SBATCH --job-name=saedit-case
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=saedit-%j.log
# Supply site-specific --partition/--qos at submission; edit no account credentials here.
set -euo pipefail
: "${SAEDIT_ROOT:?Set SAEDIT_ROOT to your pinned upstream checkout}"
cd "$SAEDIT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1
python -u examples/run_case.py "$@"
