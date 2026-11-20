#!/usr/bin/env bash
#
# Run the plan files of a crash-scene plan set in AlpaSim and check the outcome.
#
# Each plan file (control, round_NN/crash, round_NN/near_miss) is one headless,
# CPU-only AlpaSim run driven by the scripted driver. Finished runs are skipped,
# so the script can be re-run after an interruption. See docs/CRASH_SCENES.md.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: tools/run_crash_scenes.sh --plan-set DIR [options]

Options:
  --plan-set DIR     Output directory of `python -m crash_scenes plans` (required)
  --runs-dir DIR     Where run directories go (default: <plan set>/runs)
  --base-image PATH  AlpaSim image (default: $BASE_IMAGE or the 0.134.0 sandbox)
  --only NAME        Run a single plan, e.g. control or round_00/crash (repeatable)
  --dry-run          Write each run's run.sh without starting it
  -h, --help         Show this help
USAGE
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN_SET=""
RUNS_DIR=""
BASE_IMAGE="${BASE_IMAGE:-/project/AV/buc9hh/alpasim/sif-cache/alpasim-base_0.134.0-f959673709b572f9.sandbox}"
ONLY=()
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan-set) PLAN_SET="$2"; shift 2 ;;
    --runs-dir) RUNS_DIR="$2"; shift 2 ;;
    --base-image) BASE_IMAGE="$2"; shift 2 ;;
    --only) ONLY+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$PLAN_SET" ]]; then
  usage >&2
  exit 1
fi

cd "$REPO_ROOT"
PLAN_SET="$(realpath "$PLAN_SET")"
RUNS_DIR="$(realpath -m "${RUNS_DIR:-$PLAN_SET/runs}")"
PYTHON="$REPO_ROOT/.venv/bin/python"
export PYTHONPATH="$REPO_ROOT/src/utils:$REPO_ROOT/src/tools"

# The search and the rollouts must agree on these, or the mined crash times
# and actor sets no longer describe the rollout.
mapfile -t SEARCH_OVERRIDES < <("$PYTHON" - "$PLAN_SET/search_config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
for key in ("n_sim_steps", "force_gt_duration_us", "control_timestep_us", "min_traffic_duration_us"):
    print(f"runtime.simulation_config.{key}={config[key]}")
PY
)

if [[ ${#ONLY[@]} -gt 0 ]]; then
  PLAN_NAMES=("${ONLY[@]}")
else
  mapfile -t PLAN_NAMES < <(cd "$PLAN_SET" && find . -name plan.json -not -path "./runs/*" \
    | sed -e 's|^\./||' -e 's|/plan.json$||' | sort)
fi

RUN_DIRS=()
FAILED=()
for name in "${PLAN_NAMES[@]}"; do
  plan_file="$PLAN_SET/$name/plan.json"
  run_dir="$RUNS_DIR/${name//\//_}"
  RUN_DIRS+=("$run_dir")

  scene_ids="$("$PYTHON" -m crash_scenes scene-ids "$plan_file")"
  n_scenes="$(tr ',' '\n' <<< "$scene_ids" | wc -l)"
  n_done=0
  if [[ -d "$run_dir/rollouts" ]]; then
    n_done="$(find "$run_dir/rollouts" -name _complete | wc -l)"
  fi
  if [[ "$n_done" -ge "$n_scenes" ]]; then
    echo "[$name] all $n_scenes rollouts complete, skipping"
    continue
  fi

  mkdir -p "$run_dir"
  cp "$plan_file" "$run_dir/plan.json"

  cmd=(
    "$REPO_ROOT/.venv/bin/alpasim_wizard"
    deploy=local_apptainer
    topology=cpu_headless
    driver=scripted
    "defines.base_image=$BASE_IMAGE"
    "wizard.log_dir=$run_dir"
    "scenes.scenes_csv=[$PLAN_SET/scenes.csv]"
    "scenes.scene_ids=[$scene_ids]"
    runtime.enable_autoresume=true
    "${SEARCH_OVERRIDES[@]}"
  )
  if [[ "$DRY_RUN" -eq 1 ]]; then
    cmd+=(wizard.dry_run=true)
  fi

  echo "[$name] $n_scenes scenes -> $run_dir"
  if ! "${cmd[@]}"; then
    # Keep going: one failed run should not cost the rest of the plan set.
    echo "[$name] run failed; see $run_dir/txt-logs" >&2
    FAILED+=("$name")
  fi
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  "$PYTHON" -m crash_scenes confirm --plan-set "$PLAN_SET" "${RUN_DIRS[@]}"
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed runs: ${FAILED[*]}" >&2
  exit 1
fi
