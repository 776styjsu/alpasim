#!/bin/bash
#SBATCH --job-name=extract_frames
#SBATCH --partition=standard
#SBATCH --account=elbaumhpc
#SBATCH --output=logs/extract_frames_%A_%a.out
#SBATCH --error=logs/extract_frames_%A_%a.err
#SBATCH --array=0-99
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

REPO_DIR=/project/AV/alpasim

# Rollout batch to extract. Inputs are read from "${BATCH}/*/rollouts/**/rollout.asl"
# and frames are written to "${BATCH}/extracted_frames/<clip>/" so the output stays
# discoverable by the tutorial_alpamayo*/extracted_frames/*/<clip_id> resolver.
# The 0-99 array covers the 100-scene tutorial batches; tasks above the number of
# completed rollouts exit early below.
# BATCH=tutorial_alpamayo_monitor
BATCH=tutorial_alpamayo/tutorial_alpamayo15_next50_after_first50_retry

cd "${REPO_DIR}"

export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="${SLURM_TMPDIR:-/tmp}/alpasim-extract-frames-${SLURM_ARRAY_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}"

# Build sorted array of all rollout ASL files in this batch.
mapfile -t ASL_FILES < <(find "${BATCH}"/*/rollouts -name rollout.asl | sort)

TOTAL=${#ASL_FILES[@]}
if [[ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL} ]]; then
    echo "Array index ${SLURM_ARRAY_TASK_ID} out of range (total: ${TOTAL}). Exiting."
    exit 0
fi

ASL="${ASL_FILES[${SLURM_ARRAY_TASK_ID}]}"

# Derive per-rollout output dir. CLIP_DIR is .../rollouts/<clipgt-uuid>, so CLIP_NAME
# is the clipgt-<uuid> the extractor fills with per-frame <clip_id>_tNNNN subdirs.
RUN_DIR="$(dirname "${ASL}")"
CLIP_DIR="$(dirname "${RUN_DIR}")"
CLIP_NAME="$(basename "${CLIP_DIR}")"
OUTPUT_DIR="${REPO_DIR}/${BATCH}/extracted_frames/${CLIP_NAME}"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}/${TOTAL}: processing ${ASL}"
echo "  Output: ${OUTPUT_DIR}"

mkdir -p logs "${OUTPUT_DIR}"

uv run --extra eval python -m alpasim_utils.extract_frame \
    "${ASL}" \
    --output-dir "${OUTPUT_DIR}"

echo "[$(date)] Done: ${ASL}"
