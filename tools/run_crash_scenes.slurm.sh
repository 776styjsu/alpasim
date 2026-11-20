#!/usr/bin/env bash
#
# Submit with:
#   sbatch --export=ALL,PLAN_SET=experiments/crash_scenes/<plan set> tools/run_crash_scenes.slurm.sh
#
# Runs every plan file of the plan set in AlpaSim (CPU only, no renderer) and
# writes <plan set>/confirmation.jsonl. Extra arguments for
# tools/run_crash_scenes.sh go in RUN_ARGS, e.g. RUN_ARGS="--only control".

#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --mem=24G
#SBATCH -J "crash_scenes"
#SBATCH --account=elbaumhpc
#SBATCH --output=experiments/logs/crash_scenes_%j.out
#SBATCH --error=experiments/logs/crash_scenes_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/project/AV/alpasim}"
: "${PLAN_SET:?Set PLAN_SET to the output directory of 'python -m crash_scenes plans'}"

cd "$REPO_ROOT"

if ! type module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
module load apptainer

read -r -a run_args <<< "${RUN_ARGS:-}"

echo "[$(date)] Running crash scenes for $PLAN_SET on $(hostname)"
tools/run_crash_scenes.sh --plan-set "$PLAN_SET" "${run_args[@]}"
echo "[$(date)] Done."
