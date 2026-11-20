# Reasoning-coverage experiment archive — 2026-09-06

This archive preserves detailed results from experiments preceding the planned
reasoning-coverage redesign. The active
[methodology reference](../../../reasoning_coverage/REASONING_COVERAGE_METHODOLOGY.md),
[Table-based definitions](../../../reasoning_coverage/REASONING_COVERAGE_DEFINITIONS.md),
[pilot methodology](../../../reasoning_coverage/REASONING_COVERAGE_PILOT.md), and
[pre-Table definitions](../../../reasoning_coverage/REASONING_COVERAGE_DEFINITIONS_DEPRECATED.md)
remain in `docs/`.

## Historical write-ups

- [Definitions and evidence](REASONING_COVERAGE_DEFINITIONS.md)
- [Pilot and exploratory follow-ups](REASONING_COVERAGE_PILOT.md)
- [Cross-cohort experiment analysis](REASONING_COVERAGE_EXPERIMENT_ANALYSIS.md)
- [Multi-round prioritization results](REASONING_COVERAGE_MULTIROUND.md)
- [Early and unique fault discovery](REASONING_COVERAGE_EARLY_UNIQUE_FAULTS.md)

These are complete historical write-ups; only relative Markdown links were
adjusted for relocation. Statements about “current” results or the “next”
experiment describe that historical work, not the planned redesign. Commands
and paths inside code blocks retain their original spelling.

## Generated artifacts and audit records

| Original location | Archived location |
| --- | --- |
| `runs/reasoning_coverage/` | [Reports, caches, labels, CSVs, HTML, and figures](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/) |
| `data/reasoning_coverage/` | [Coarse-label and program-parser audits](../../../../experiments/archive/reasoning_coverage/2026-09-06/data/reasoning_coverage/) |

The [manifest](../../../../experiments/archive/reasoning_coverage/2026-09-06/manifest.json)
records every artifact's original path, archived path, size, and SHA-256.
Artifact bytes, including embedded provenance paths and hashes, are unchanged.
The artifact archive is local and gitignored through `/experiments/`, as are
other experiment outputs. Source code, tests, shared benchmark JSONs, and
root-level rollout inputs remain in their existing locations.

## Reproduction after relocation

Run from the repository root. Existing CLI defaults still use the old `runs/`
paths; explicitly pass archived input paths and a fresh output directory.
For the cross-cohort analysis:

```bash
coverage_archive=experiments/archive/reasoning_coverage/2026-09-06
coverage_inputs="$coverage_archive/runs/reasoning_coverage"
coverage_output=experiments/results/reasoning_coverage_reproduction

PYTHONPATH=src/tools uv run python -m reasoning_coverage.experiment_analysis \
  --planned-labels "$coverage_inputs/ood_hazard_2604.planned_safety_3s_100ms.json" \
  --feature-cache "$coverage_inputs/ood_hazard_2604.features.json" \
  --exposures "$coverage_inputs/ood_hazard_2604.cro_v1_exposures.json" \
  --output "$coverage_output/experiment_analysis.json" \
  --figure-dir "$coverage_output/figures" \
  --benchmark-repeats 200 --planned-repeats 50 --workers 16
```

Add `--multi-round` and choose distinct output/figure paths for that comparison.
The safety evaluator uses the same labels and feature cache:

```bash
PYTHONPATH=src/tools uv run python -m reasoning_coverage.planned_safety_evaluation \
  --labels "$coverage_inputs/ood_hazard_2604.planned_safety_3s_100ms.json" \
  --feature-cache "$coverage_inputs/ood_hazard_2604.features.json" \
  --output "$coverage_output/planned_safety_coverage.json"

PYTHONPATH=src/tools uv run python -m reasoning_coverage.table_evaluation \
  --benchmark-output "$coverage_output/benchmark150.table_program_coverage.json" \
  --frame-output "$coverage_output/ood_hazard_2604.frames.table_program_coverage.json"
```

For the pilot's audited variant, supply
`--audit-labels "$coverage_archive/data/reasoning_coverage/codex_audit_v1.json"`.
For other historical commands, map old artifact inputs using the table above
and direct outputs to a fresh directory. Original external OOD rollout and
unsafe-event inputs are still required by the relevant evaluators; their
locations are documented in the historical pilot and CLI arguments.
