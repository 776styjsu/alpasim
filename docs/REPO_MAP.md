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
| `tools/` | yes | Standalone scripts and job launchers. Not an importable package; run with `python tools/<name>.py`. |
| `docs/` | yes | All prose documentation, including this file. |
| `plugins/` | yes | Out-of-tree driver plugins (`alpasim_transfuser`, `transfuser_driver`). Workspace members via `plugins/*`. |
| `data/` | partly | Scene assets, driver checkpoints, NRE artifacts, HF caches. ~494 GB; almost entirely gitignored via `data/.gitignore`. |
| `experiments/` | **no** | All local experiment output. See [Experiment data](#experiment-data). |
| `SafeVLA/` | **no** | LaTeX sources and dated snapshot zips for the *Drive the Thoughts* paper. |
| `tutorial_alpamayo*`, `benchmark*.json` | **no** | Load-bearing experiment inputs that must stay at the root. See [Root-level exceptions](#root-level-exceptions-do-not-move-these). |
| `logs/`, `outputs/`, `runs/` | **no** | Default output directories for the wizard, Hydra and the Slurm runner. Recreated automatically; safe to empty. |
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
| `alpasim_driver` | `src/driver` | Adapters for ego policies (VAVAM, Alpamayo 1 / 1.5, …). |
| `alpasim_eval` | `src/eval` | Metric accumulation and rollout evaluation. |
| `alpasim_utils` | `src/utils` | Shared helpers: artifacts, ASL log reading, frame extraction, geometry, lane projection. |
| `utils_rs` | `src/utils_rs` | Rust acceleration for `alpasim_utils`. |
| `alpasim_plugins` | `src/plugins` | Plugin registry and discovery. |
| `alpasim-tools` | `src/tools` | Analysis toolkit. See below. |

### Analysis toolkit (`src/tools/`)

This is where the research line lives. Modules are importable
(`PYTHONPATH=src/utils:src/tools`), and most expose a `__main__`.

| Module | Purpose |
| --- | --- |
| `cot_analysis/` | CoT consistency analysis over simulation outputs — the F-LLM judge pipeline. |
| `reasoning_coverage/` | Coverage abstractions and prioritization for reasoning traces: `program.py` / `table_program.py` compile traces, `analysis.py` builds coverage sets and APFD, `statistics.py` runs permutation/bootstrap tests, and `*_evaluation.py` are the per-experiment drivers (`obligation`, `ood`, `planned_safety`, `table`). `experiment_analysis.py` is the top-level report generator. |
| `trajectory_safety/` | Geometric safety outcomes for planned ego trajectories (Experiment A). |
| `rollout_monitoring/` | Closed-loop online CoT monitoring analysis (Experiment B). |
| `map_utils/` | Map visualization and analysis utilities. |
| `tests/` | Pytest suite for all of the above. |
| `run-on-slurm/`, `run-on-ord/`, `scripts/` | Job submission and operational helpers. |

Reasoning-coverage methodology is retained in
[REASONING_COVERAGE_METHODOLOGY.md](REASONING_COVERAGE_METHODOLOGY.md),
[REASONING_COVERAGE_DEFINITIONS.md](REASONING_COVERAGE_DEFINITIONS.md),
[REASONING_COVERAGE_PILOT.md](REASONING_COVERAGE_PILOT.md), and
[REASONING_COVERAGE_DEFINITIONS_DEPRECATED.md](REASONING_COVERAGE_DEFINITIONS_DEPRECATED.md).
Detailed historical write-ups are in the
[reasoning-coverage archive](archive/reasoning_coverage/2026-09-06/README.md).
Generated reports, figures, caches, and audit records are under
`experiments/archive/reasoning_coverage/2026-09-06/`; the archive index maps paths
and documents reproduction commands. The redesign has a draft
[reasoning-path extraction prompt](REASONING_COVERAGE_PARSER_PROMPT.md);
its parser implementation and validation are pending.

Running the tests without disturbing the environment:

```bash
PYTHONPATH=src/utils:src/tools .venv/bin/python -m pytest src/tools/tests -q
```

> Prefer the venv interpreter directly for ad-hoc runs. `uv run` re-resolves the
> environment and has previously replaced `./.venv`; recover with
> `uv sync --extra all --extra transfuser`.

## Experiment data

Everything the research runs produce lives under `experiments/`, which is
gitignored wholesale. Layout:

| Directory | Contents |
| --- | --- |
| `experiments/rollouts/` | Wizard run directories — smoke runs and superseded `tutorial_alpamayo_*` sets. Each holds `rollouts/`, `txt-logs/`, `prometheus/` and the generated `*-config.yaml` files. |
| `experiments/results/` | Derived metrics: downstream-safety and rule-consistency JSON/CSV, coverage distributions, reliability heatmaps, NuRec label-coverage candidates. |
| `experiments/bundles/` | Annotation and rollout-video zips, plus the two extracted `benchmark_annotation_bundle_unsafe_reliable*` trees. |
| `experiments/plots/` | Generated trajectory plot directories. |
| `experiments/logs/` | Stray job and serving logs. |
| `experiments/scratch/` | One-off throwaway scripts kept only for provenance. Nothing imports these. |
| `experiments/archive/` | Older archived material, including reasoning-coverage results under `reasoning_coverage/2026-09-06/`. |
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
2. **Repo-root joins and markers.** `tools/build_artifact_release.py` and
   `cot_analysis/__main__.py` join or pattern-match against the literal names
   `tutorial_alpamayo` and `tutorial_alpamayo15_first20`.

What that pins to the root:

- **Rollout directories** referenced by `benchmark.json`: `tutorial_alpamayo`,
  `tutorial_alpamayo15_first20`, `tutorial_alpamayo_next100_unique_flat`,
  `tutorial_alpamayo_next100_2_unique_flat`, `tutorial_alpamayo_upstream_2601`.
- **Benchmark JSONs whose entries carry `source_scene_file`**: `benchmark.json`
  (also the default `--benchmark` value in seven modules),
  `benchmark_expanded_100.json`, `benchmark_pre_calibration.json`,
  `benchmark_old.json`, and the twelve `benchmark_expanded_100.cot_consistency*.json`
  monitor outputs.

`benchmark.json` also names rollout roots that have no directory of their own at
the top level (`tutorial_alpamayo15_first50`, `tutorial_alpamayo15_coverage_boost*`).
These resolve through the second candidate, `root / "tutorial_alpamayo" / path` —
they are nested inside `tutorial_alpamayo/`. That fallback is why the literal name
`tutorial_alpamayo` is load-bearing well beyond its own contents.

Verified after the 2026-08-29 reorganization: all 150 `source_scene_file` entries
in `benchmark.json` still resolve. Re-run that check after any future move:

```bash
PYTHONPATH=src/utils:src/tools .venv/bin/python -c "
import json; from pathlib import Path
root = Path('benchmark.json').resolve().parent
d = json.load(open(root/'benchmark.json'))
es = d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list))
ss = [e['source_scene_file'] for e in es if isinstance(e, dict) and e.get('source_scene_file')]
bad = [s for s in ss if not ((root/s).exists() or (root/'tutorial_alpamayo'/s).exists())]
print(f'{len(ss)-len(bad)} resolved, {len(bad)} unresolved')"
```

Some analysis modules additionally point at a sibling working copy at
`/project/AV/drive-the-thoughts/alpasim/` (for `tutorial_alpamayo_ood_hazard_2604`),
which is outside this repository.

**To decouple properly** (worth doing before the next reorganization): add an
explicit `--scene-root` argument to the evaluation entry points and thread it
through the `_resolve_scene_dir` helpers, instead of deriving roots from the
benchmark file's location.

## Where new things go

| If you are adding… | Put it in… |
| --- | --- |
| An importable analysis module | `src/tools/<area>/` |
| A test for one | `src/tools/tests/test_<area>_*.py` |
| A launcher, plotter or one-shot runner | `tools/` |
| Prose, a definition, an experiment write-up | `docs/` |
| Anything a run produces | `experiments/<category>/` — never the repo root |
| A quick throwaway probe | `experiments/scratch/`, or better, a scratch dir outside the repo |

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
