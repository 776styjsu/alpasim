#!/usr/bin/env bash
#
# Submit with:
#   sbatch --export=ALL,OUTPUT_DIR=experiments/crash_scenes/mine_<date> tools/mine_crash_scenes.slurm.sh
#
# Searches every locally cached scene for ego maneuvers that end in a crash and
# writes <OUTPUT_DIR>/manifest.jsonl and summary.json. CPU only; no simulator
# runs. See docs/CRASH_SCENES.md.

#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=standard
#SBATCH --mem=48G
#SBATCH -J "crash_mine"
#SBATCH --account=elbaumhpc
#SBATCH --output=experiments/logs/crash_mine_%j.out
#SBATCH --error=experiments/logs/crash_mine_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/project/AV/alpasim}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR, e.g. experiments/crash_scenes/mine_<date>}"
USDZ_DIR="${USDZ_DIR:-data/nre-artifacts/all-usdzs}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src/utils:$REPO_ROOT/src/tools"

echo "[$(date)] Mining $USDZ_DIR -> $OUTPUT_DIR on $(hostname)"
.venv/bin/python -m crash_scenes mine \
  --usdz-dir "$USDZ_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
echo "[$(date)] Done."
