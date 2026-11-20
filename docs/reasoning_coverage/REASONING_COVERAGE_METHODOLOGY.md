# Previous reasoning-coverage experiment methodology

This reference collects methods used across previous reasoning-coverage
experiments. Detailed results and original write-ups are in the
[September 2026 archive](../archive/reasoning_coverage/2026-09-06/README.md).
The new reasoning-coverage structure has not yet been defined.

## Representations and controls

The [Table-based definitions](REASONING_COVERAGE_DEFINITIONS.md) describe Z1
components, Z2 support relations, Z3 decisions, and explicit continuations.
The [pilot methodology](REASONING_COVERAGE_PILOT.md) and
[pre-Table definitions](REASONING_COVERAGE_DEFINITIONS_DEPRECATED.md) retain the
coarse-state, compiled-program, and causal-obligation approaches.

The full primary-stage representation accumulates nodes, details, edges, and
complete typed paths. The atomic ablation keeps only complete
`Z1 component → Z2 role → Z3 axis:decision` paths. One trace can contribute
several atomic paths, for example when both a lead vehicle and an available
gap justify a lane change. Atomic coverage retains Z2 relations.

Compare reasoning with random ordering, NuRec specification coverage, visual
statistics, image-grid coverage, trajectory summaries, ego-centric trajectory
grids, and text bigrams where the required inputs exist. Fit quantile bins
without outcomes and report each arm's fixed capacity or observed universe.

## Ordering and saturation

Greedy ordering selects the remaining case with the largest uncovered-item
gain. Randomized ties use recorded seeds shared across comparisons.

- Single-round ordering uses the seeded tie order after coverage saturates.
- Multi-round ordering clears the covered set when remaining cases provide no
  additional coverage, then resumes greedy selection. Selected cases remain
  removed; later rounds cover the items available in the remaining cases.

Report first-round saturation in cases and as a fraction of the cohort. Hold
coverage construction, cohorts, budgets, seeds, and repeat counts fixed when
comparing the two ordering rules. Previous cross-cohort studies used 200
benchmark repeats and 50 planned-safety repeats.

## Outcomes and uncertainty

Keep outcomes hidden during compilation and ordering. Declare the test and
fault units, valid/excluded cohort, source-scene overlap policy, proxy horizon,
and review budgets. Report APFD alongside recall or positive cases detected at
fixed budgets: whole-order APFD can conceal poor performance at early budgets.

Positive frames from the same scene are dependent. Frame counts weight a
persistent failure by its duration. Claims about distinct failures require an
event definition and event-level evaluation. Tie-breaking intervals describe
ordering variation; source-scene resampling addresses a different uncertainty.

## Early and unique discoveries

Compare detected fault sets at the same stated budget, paired by tie repeat:

1. **Pairwise:** report candidate-only, shared, and comparator-only faults when
   each arm receives the same budget.
2. **Non-reasoning union:** compare against the union of non-reasoning arms,
   each receiving the full budget. This is stricter than a portfolio sharing
   one budget; list the included comparators for each cohort.
3. **Robust exclusivity:** previous studies required detection in at least
   75% of reasoning repeats and at most 25% of comparator-union repeats.

“Unique” describes discovery under these orderings and budgets. It does not
establish that another representation cannot discover the fault.

## Reproduction

The [archive index](../archive/reasoning_coverage/2026-09-06/README.md) maps original
artifact paths to archived locations and provides commands that read archived
inputs while writing fresh output under `experiments/results/`.
