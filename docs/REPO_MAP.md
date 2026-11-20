# Repository map

Orientation for anyone — human or agent — landing in this working copy for the
first time. [AGENTS.md](../AGENTS.md) covers *how* to build, test and contribute;
this file covers *where things are* and *why they sit where they do*.

This is a **research working copy**, not a clean checkout. Alongside the upstream
AlpaSim simulator it carries a chain-of-thought reliability / reasoning-coverage
research line, several hundred GB of rollout data, and a LaTeX paper. The rules
below exist so that clutter does not creep back.

## Top-level layout

| Entry | Tracked | What it is |
| --- | --- | --- |
| `src/` | yes | The uv workspace — every Python/Rust package. See below. |
| `tools/` | yes | Standalone research scripts and Slurm launchers. Not an importable package; run from the repo root as `python tools/<name>.py` or `sbatch tools/<name>.slurm.sh`. See [tools index](#research-scripts-tools). |
| `docs/` | yes | All prose documentation, including this file. Research write-ups are in `docs/reasoning_coverage/`; superseded ones in `docs/archive/`. |
| `plugins/` | yes | Out-of-tree driver plugins (`alpasim_transfuser`, `transfuser_driver`). Workspace members via `plugins/*`. |
| `data/` | partly | Scene assets, driver checkpoints, NRE artifacts, HF caches. ~494 GB; almost entirely gitignored via `data/.gitignore`. `data/reasoning_coverage/decision_role_proposition_v1.json` is the tracked proposition vocabulary read by `reasoning_coverage.parsed_evaluation`. |
| `Dockerfile`, `Apptainer.def`, `build_apptainer.sh` | yes | Container builds. The Apptainer pair comes from the `apptainer-support` branch and sits beside `Dockerfile` deliberately; `build_apptainer.sh` writes images to `sif-cache/`. See [APPTAINER.md](APPTAINER.md). |
| `AGENTS.md`, `CLAUDE.md`, `CONTEXT.md` | yes | Agent instructions (`CLAUDE.md` just includes `AGENTS.md`) and the upstream domain glossary (Run UUID vs Run Name vs NRE Run ID, …). |
| `.agents/`, `.cursor/`, `.claude/` | partly | Agent skills and editor rules. `.claude/` is local-only. |
| `experiments/` | **no** | All local experiment output, including Slurm logs. See [Experiment data](#experiment-data). |
| `SafeVLA/` | **no** | LaTeX sources and dated snapshot zips for the *Drive the Thoughts* paper. |
| `tutorial_alpamayo*`, `benchmark*.json` | **no** | Load-bearing experiment inputs that must stay at the root. See [Root-level exceptions](#root-level-exceptions-do-not-move-these). |
| `runs/` | **no** | **Not disposable.** `runs/llm_matrix/` is the judge-matrix result set behind the paper's repeated-run evaluation — the default input of `tools/plot_repeated_main_eval.py` and `tools/build_artifact_release.py`, and the live directory updated by `tools/run_benchmark_llm_matrix_update.slurm.sh`. The `llm_matrix_*` siblings are earlier subsets/variants. `runs/slurm_output/` is written by the upstream `src/tools/run-on-slurm/submit.sh`. |
| `outputs/` | **no** | Hydra's per-invocation output directory (`outputs/<date>/<time>/`). Recreated automatically; safe to empty. |
| `sif-cache/`, `.hf-cache/`, `.uv-cache/`, `.venv/`, `.qwen*-vllm-env/` | **no** | Build/model/dependency caches and virtualenvs. ~175 GB combined. |

## The workspace (`src/`)

Members are declared in the root `pyproject.toml` under `[tool.uv.workspace]`.
Install a single one with the matching extra, e.g. `uv sync --extra runtime`.

| Package | Path | Role |
| --- | --- | --- |
| `alpasim_wizard` | `src/wizard` | Entry point. Composes a run from Hydra configs in `src/wizard/configs/` (groups: `deploy`, `topology`, `driver`, `physics`, `controller`, `trafficsim`, `cameras`, `vehicle`, `exp`, …) and launches the services. |
| `alpasim-runtime` | `src/runtime` | Orchestrates the simulation loop across services. |
| `alpasim_grpc` | `src/grpc` | Protobuf/gRPC API definitions. Regenerate with `cd src/grpc && uv run compile-protos` after editing a `.proto`. |
| `alpasim-physics` | `src/physics` | Physics micro-service. |
| `alpasim-trafficsim` | `src/trafficsim` | Traffic micro-service. |
| `alpasim_controller` | `src/controller` | Controller and vehicle model. |
| `alpasim_driver` | `src/driver` | Adapters for ego policies (VAVAM, Alpamayo 1 / 1.5 / 2, …). |
| `alpasim_eval` | `src/eval` | Metric accumulation and rollout evaluation. |
| `alpasim_utils` | `src/utils` | Shared helpers: artifacts, ASL log reading, frame extraction, geometry, lane projection, CoT consistency. |
| `utils_rs` | `src/utils_rs` | Rust acceleration for `alpasim_utils`. |
| `alpasim_plugins` | `src/plugins` | Plugin registry and discovery. |
| `alpasim-tools` | `src/tools` | Analysis toolkit. See below. |

### Analysis toolkit (`src/tools/`)

This is where the research line lives. Modules are importable
(`PYTHONPATH=src/utils:src/tools`), and most expose a `__main__`.

| Module | Purpose |
| --- | --- |
| `cot_analysis/` | CoT consistency analysis over simulation outputs — the F-LLM judge pipeline. `prompt*.py` are the judge prompt versions (`*_outdated.py` are kept for reproducing old runs). |
| `reasoning_coverage/` | Coverage abstractions and prioritization for reasoning traces: `program.py` / `table_program.py` compile traces, `trajectory_grid.py` is the trajectory baseline, `analysis.py` builds coverage sets and APFD, `statistics.py` runs permutation/bootstrap tests, and `*_evaluation.py` are the per-experiment drivers (`obligation`, `ood`, `planned_safety`, `table`, `parsed`). `parsed_paths.py` builds coverage arms from LLM parses; `temporal_sensitivity.py` probes frame-timing robustness. `experiment_analysis.py` is the top-level report generator. |
| `trajectory_safety/` | Geometric safety outcomes for planned ego trajectories (Experiment A). |
| `rollout_monitoring/` | Closed-loop online CoT monitoring analysis (Experiment B). |
| `crash_scenes/` | Forced-crash scenes for testing safety monitors: `label.py` labels what the ego hit and where, `mine.py` searches recorded scenes for ego maneuvers that crash, `plans.py` writes the plan files that `scripted_driver.py` follows in AlpaSim, `confirm.py` checks the outcome, `monitor_replay.py` runs a log-driven monitor over a saved rollout. See [CRASH_SCENES.md](CRASH_SCENES.md). |
| `map_utils/` | Map visualization and analysis utilities. |
| `tests/` | Pytest suite for all of the above. |
| `run-on-slurm/`, `run-on-ord/`, `scripts/` | Upstream job submission and operational helpers. |

Reasoning-coverage documentation lives in
[docs/reasoning_coverage/](reasoning_coverage/):
[methodology](reasoning_coverage/REASONING_COVERAGE_METHODOLOGY.md),
[definitions](reasoning_coverage/REASONING_COVERAGE_DEFINITIONS.md),
[pilot](reasoning_coverage/REASONING_COVERAGE_PILOT.md), and the retained
[pre-Table definitions](reasoning_coverage/REASONING_COVERAGE_DEFINITIONS_DEPRECATED.md).
Detailed historical write-ups are in the
[reasoning-coverage archive](archive/reasoning_coverage/2026-09-06/README.md).
Generated reports, figures, caches, and audit records are under
`experiments/archive/reasoning_coverage/2026-09-06/`; the archive index maps paths
and documents reproduction commands. The redesign has a draft
[reasoning-path extraction prompt](reasoning_coverage/REASONING_COVERAGE_PARSER_PROMPT.md) and a
[parsed-path prioritization protocol](reasoning_coverage/REASONING_COVERAGE_PARSED_PRIORITIZATION.md).
`reasoning_coverage/parsed_paths.py` constructs the five coverage arms from
existing parses; `parsed_evaluation.py` compares them with trajectory-grid and
random ordering on the large cohort. Results are in
`experiments/results/reasoning_coverage_prioritization_2026-09-08/`. Human
validation of parser accuracy is still pending.

Running the tests without disturbing the environment:

```bash
PYTHONPATH=src/utils:src/tools .venv/bin/python -m pytest src/tools/tests -q
```

> Prefer the venv interpreter directly for ad-hoc runs. `uv run` re-resolves the
> environment and has previously replaced `./.venv`; recover with
> `uv sync --extra all --extra transfuser`.

### Research scripts (`tools/`)

All assume the repository root as working directory. Slurm launchers write their
stdout/stderr to `experiments/logs/`.

| Purpose | Scripts |
| --- | --- |
| Forced-crash scenes | `mine_crash_scenes.slurm.sh` (search scenes for crash maneuvers), `run_crash_scenes.sh` / `.slurm.sh` (run a plan set headless and confirm it) |
| Run Alpamayo rollouts | `run_top_scenes_alpamayo.sh`, `run_top_scenes_alpamayo.slurm.sh`, `run_top_scenes_alpamayo_monitor.slurm.sh` (with the online consistency monitor) |
| Post-process rollouts | `extract_rollout_frames.sh` (one run dir), `extract_frames.slurm.sh` (array job over a batch), `build_flat_tutorial.py` (symlink-aggregate per-scene runs into one `*_flat` dir) |
| CoT judge runs | `run_tutorial_alpamayo_cot_analysis.py` / `.slurm.sh`, `run_benchmark_llm_matrix.py`, `run_benchmark_llm_matrix_update.slurm.sh`, `serve_qwen3*_4b_fp8_vllm.sh` (local judge server) |
| Judge accuracy and reports | `check_consistency_accuracy.py`, `build_consistency_spreadsheet.py`, `build_benchmark_comparison.py`, `generate_benchmark_reasoning_signatures_html.py`, `export_benchmark_frame_folders.py` |
| Figures | `plot_cot_reliability_heatmap.py`, `plot_repeated_main_eval.py`, `plot_trajectory_grid_path_histogram.py` |
| Paper artifact | `build_artifact_release.py` (stages the *Drive the Thoughts* release) |
| Repo hygiene | `precommit/` (license-header hook) |

## Experiment data

Everything the research runs produce lives under `experiments/`, which is
gitignored wholesale. Layout:

| Directory | Contents |
| --- | --- |
| `experiments/rollouts/` | Wizard run directories — smoke runs (`a15_smoke*`, `a2_smoke*`) and superseded `tutorial_alpamayo_*` sets. Each holds `rollouts/`, `txt-logs/`, `prometheus/` and the generated `*-config.yaml` files. **Symlink targets** — see below. |
| `experiments/results/` | Derived metrics: downstream-safety and rule-consistency JSON/CSV, coverage distributions, reliability heatmaps, NuRec label coverage. Dated subdirectories (`*_YYYY-MM-DD`, `*_YYYYMMDD`) hold one analysis each. |
| `experiments/cot_parse/` | LLM reasoning-path parses: `batches/` (inputs) and `out/` (parser outputs). Default `--parse-inputs` / `--parse-outputs` of `reasoning_coverage.parsed_evaluation`. |
| `experiments/crash_scenes/` | Output of `crash_scenes`: `mine_<date>/` (manifest of planned crashes), `plans_<name>/` (plan files, their `runs/` and `confirmation.jsonl`), `phase1_validation/` (labeller and monitor-replay checks on older rollouts). |
| `experiments/bundles/` | Annotation and rollout-video zips, plus the two extracted `benchmark_annotation_bundle_unsafe_reliable*` trees. |
| `experiments/plots/` | Generated plot directories. |
| `experiments/logs/` | Slurm job logs from `tools/*.slurm.sh`, model-serving logs, and older stray logs (`archive_0614/`, `archive_0628/`). |
| `experiments/scratch/` | One-off analysis and probe scripts kept only for provenance (dated in the filename). Nothing imports these; promote anything reusable into `src/tools/` or `tools/`. |
| `experiments/archive/` | Superseded results, including reasoning-coverage results under `reasoning_coverage/2026-09-06/`, `benchmark.json.bak_odd_constraints`, and the `symlink_retarget_2026-11-11.tsv` manifest. |
| `experiments/_to_delete/` | Empty directories, zero-byte files and regenerable caches. Nothing here is referenced; delete freely. |

### Root-level exceptions (do not move these)

Several artifacts **must** remain at the repository root, because the analysis
code resolves scene paths relative to them. Moving them causes *silent* failure —
scene lookups return `None` and results are quietly incomplete rather than
erroring. Two distinct mechanisms:

1. **Sibling resolution.** `reasoning_coverage/analysis.py`,
   `reasoning_coverage/table_evaluation.py`, `trajectory_safety/__main__.py` and
   `tools/export_benchmark_frame_folders.py` resolve a benchmark entry's
   `source_scene_file` as `benchmark_path.resolve().parent / <rollout_dir> / …`.
   The benchmark JSON and the rollout directories it names therefore have to be
   siblings. Note `.resolve()` — a symlink does not help, it resolves to the real
   parent.
2. **Repo-root joins and defaults.** `tools/build_artifact_release.py` and
   `cot_analysis/__main__.py` join or pattern-match against the literal names
   `tutorial_alpamayo` and `tutorial_alpamayo15_first20`, and fourteen entry
   points in `src/tools/reasoning_coverage/` and `tools/` default to
   `benchmark.json` at the repo root.

What that pins to the root:

- **Rollout directories** referenced by `benchmark.json`: `tutorial_alpamayo`,
  `tutorial_alpamayo15_first20`, `tutorial_alpamayo_next100_unique_flat`,
  `tutorial_alpamayo_next100_2_unique_flat`, `tutorial_alpamayo_upstream_2601`.
- **Benchmark JSONs whose entries carry `source_scene_file`**: `benchmark.json`,
  `benchmark_expanded_100.json`, `benchmark_pre_calibration.json`,
  `benchmark_old.json`, and the eleven `benchmark_expanded_100.cot_consistency*.json`
  monitor outputs (the `map_graph_gpt55*` ones are the paper's Table IV source).

`benchmark.json` also names rollout roots that have no directory of their own at
the top level (`tutorial_alpamayo15_first50`, `tutorial_alpamayo15_coverage_boost*`).
These resolve through the second candidate, `root / "tutorial_alpamayo" / path` —
they are nested inside `tutorial_alpamayo/`. That fallback is why the literal name
`tutorial_alpamayo` is load-bearing well beyond its own contents.

**Absolute symlinks.** The two `*_flat` directories are built by
`tools/build_flat_tutorial.py` from **absolute** symlinks into
`experiments/rollouts/tutorial_alpamayo_next100_unique/` and
`…_next100_2_unique/` (their `rollouts/`, `controller/`, `aggregate/` and config
files). Moving or renaming those two rollout directories breaks every link
without any error, and the `source_scene_file` check below does not notice,
because the benchmark points at real files under `extracted_frames/`. This is
exactly what happened in the 2026-08-29 reorganization (fixed 2026-11-11).

Re-run both checks after any move:

```bash
PYTHONPATH=src/utils:src/tools .venv/bin/python -c "
import json; from pathlib import Path
root = Path('benchmark.json').resolve().parent
d = json.load(open(root/'benchmark.json'))
es = d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list))
ss = [e['source_scene_file'] for e in es if isinstance(e, dict) and e.get('source_scene_file')]
bad = [s for s in ss if not ((root/s).exists() or (root/'tutorial_alpamayo'/s).exists())]
print(f'{len(ss)-len(bad)} resolved, {len(bad)} unresolved')"

# Expect 43: dangling ../../all/*.mp4 links in tutorial_alpamayo/tutorial_alpamayo_1_old
# whose videos no longer exist. Anything above that is new breakage.
find tutorial_alpamayo* -xtype l | wc -l
```

Some analysis modules additionally point at a sibling working copy at
`/project/AV/drive-the-thoughts/alpasim/` (for `tutorial_alpamayo_ood_hazard_2604`),
which is outside this repository.

**To decouple properly** (worth doing before the next reorganization): add an
explicit `--scene-root` argument to the evaluation entry points and thread it
through the `_resolve_scene_dir` helpers, instead of deriving roots from the
benchmark file's location; then move the benchmark JSONs and rollout directories
together under `experiments/`, and rebuild the `*_flat` links.

## Where new things go

| If you are adding… | Put it in… |
| --- | --- |
| An importable analysis module | `src/tools/<area>/` |
| A test for one | `src/tools/tests/test_<area>_*.py` |
| A launcher, plotter or one-shot runner | `tools/` — never the repo root |
| A Slurm script | `tools/<name>.slurm.sh`, with `#SBATCH --output=experiments/logs/<name>_%j.out` |
| Prose, a definition, an experiment write-up | `docs/` (reasoning-coverage material in `docs/reasoning_coverage/`) |
| Anything a run produces | `experiments/<category>/` — never the repo root |
| A quick throwaway probe | `experiments/scratch/<name>_<YYYYMMDD>.py`, or better, a scratch dir outside the repo |

Do not leave generated output at the repository root. The `.gitignore` rules
`/tutorial_alpamayo*` and `/benchmark*` are deliberately narrow so that the
grandfathered root files stay ignored; a new stray file with any other name will
show up as untracked and add noise to every `git status`.

## Cleanup log

### 2026-08-29 — root reorganization

The root had accumulated ~120 entries. Introduced `experiments/` and relocated 62
of them; **nothing was deleted**, every move was a same-filesystem rename.

- Moved to `experiments/rollouts/`: `a15_smoke`, `a15_smoke_2604`,
  `tutorial_alpamayo_bottom100_old`, `tutorial_alpamayo_labeled22_upstream_2505`,
  `tutorial_alpamayo_monitor`, `tutorial_alpamayo_monitor_gpt`,
  `tutorial_alpamayo_next100_2_unique`, `tutorial_alpamayo_next100_unique`,
  `tutorial_alpamayo_no_monitor`, `tutorial_archive` (~200 GB).
- Moved `archive/` → `experiments/archive/`; six `*.zip` bundles and their two
  extracted trees → `experiments/bundles/`; nineteen derived result files →
  `experiments/results/`; two plot directories → `experiments/plots/`; three
  stray logs → `experiments/logs/`.
- Moved twelve March-2026 one-off scripts (`patch*.py`, root-level `test_*.py`,
  `plot_test.py`, `sim_result_helper.py`, `test_sed.csv`) and the stray
  `src/cot_consistency.json` → `experiments/scratch/`. None were imported anywhere.
- Quarantined in `experiments/_to_delete/`: the empty `denied/`, `permission/`
  and `slurm_logs/` directories, the zero-byte `.codex` and
  `build_apptainer.sh.tmp`, and the root `__pycache__`.
- Moved `ALPAMAYO15_QUICKSTART.md` → `docs/`.
- `.gitignore`: added `/experiments/` and `/SafeVLA/`; dropped the now-dead
  `/archive/`, `/tutorial_archive/`, `/nurec_label_coverage_*`, `/result.json`
  and `/*_trajectory_plots/` rules.

Left in place deliberately: the root-level rollout directories and benchmark
JSONs listed under [Root-level exceptions](#root-level-exceptions-do-not-move-these),
and the untracked reasoning-coverage work under `src/tools/`, `docs/` and
`tools/`, which remains uncommitted on the `safevla` branch.

### 2026-09-06 — reasoning-coverage results archive

Preserved methodology in the definitions and pilot documents, retained the
pre-Table definitions, and added a cross-experiment methodology reference.
Moved the three result-focused write-ups and preserved complete versions of
the mixed definitions/pilot reports under
`docs/archive/reasoning_coverage/2026-09-06/`.

Moved `runs/reasoning_coverage/` and `data/reasoning_coverage/` into
`experiments/archive/reasoning_coverage/2026-09-06/`, preserving their directory
structure and file bytes. The archive includes a checksum manifest and updated
documentation links. No experiment artifacts were deleted. Source code, tests,
shared benchmark inputs, and root-level rollout directories were left in place.

### 2026-11-11 — root scripts, logs, docs, and broken symlinks

Nothing was deleted.

- **Repaired 391 dangling symlinks.** The 2026-08-29 move of
  `tutorial_alpamayo_next100{,_2}_unique` into `experiments/rollouts/` had broken
  all 345 absolute links in the two `*_flat` directories (their `rollouts/`,
  `controller/`, `aggregate/` and config files); these now point into
  `experiments/rollouts/`. Also repaired 46 older links in
  `tutorial_alpamayo/tutorial_alpamayo_1_old/` that still named the scene
  directories' pre-rename location. Old and new targets are recorded in
  `experiments/archive/symlink_retarget_2026-11-11.tsv`. The 43 remaining
  dangling links in `tutorial_alpamayo_1_old/` point at videos that no longer exist.
- Moved the tracked root scripts into `tools/`: `build_benchmark_comparison.py`,
  `build_flat_tutorial.py`, and `extract_frames.slurm` (now
  `tools/extract_frames.slurm.sh`). `build_benchmark_comparison.py` now falls
  back to `experiments/archive/` (its `archive/` fallback had been dead since
  2026-08-29) and writes its HTML to `experiments/results/` by default.
- Pointed every `tools/*.slurm.sh` log path (`#SBATCH --output/--error`, vLLM
  server logs, and the `mkdir -p` that pre-creates the log directory) at
  `experiments/logs/`, and moved the 286 entries of the root `logs/` there.
  The root `logs/` directory no longer exists.
- Moved the six reasoning-coverage documents into `docs/reasoning_coverage/`
  and updated every relative link to and from them.
- Moved the root `benchmark.json.bak_odd_constraints` to `experiments/archive/`.
  Like any benchmark JSON, it only resolves scenes when placed back beside the
  rollout directories.
- Corrected this map: `runs/` is not disposable (`runs/llm_matrix/` is paper
  data), and documented `experiments/cot_parse/`, the container and agent files,
  and the `tools/` scripts.
