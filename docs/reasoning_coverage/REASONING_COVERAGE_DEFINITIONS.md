# Alpamayo reasoning-coverage definitions

This document preserves the reasoning-coverage methodology used in previous
Alpamayo experiments. It is a reference for the planned redesign.
It specifies the reasoning representation, coverage obligations, denominators,
test-prioritization procedure, and evaluation protocol.

Coverage construction is outcome-blind. Collision, off-road, chain-of-thought
quality, and manual unsafe labels are used only after a coverage order has been
created.

## Reasoning abstraction

Each reasoning stage answers three questions:

```text
Z1: What matters in the scene?
Z2: How does it support or constrain the decision?
Z3: What driving behavior is chosen?
```

One reasoning output may state an immediate decision followed by explicit
continuations. It is represented as

```text
R = (S0, S1, ..., Sk)
S = (Z1, Z2, Z3, guard)
```

`S0` is the immediate reasoning stage. Later stages are compiled from explicit
continuations such as “then,” “once the path is clear,” or “when a gap opens.”

### Z1: critical components

Z1 contains the scene components claimed by the reasoning text. Every component
has:

```text
(Table 2 category, predicate, location, modality)
```

The fixed category vocabulary is the seven categories in Alpamayo Table 2:

1. critical objects;
2. traffic lights;
3. stop/yield control;
4. road events;
5. lanes/lane lines;
6. routing intent; and
7. ODD constraints.

The compiler refines these categories with predicates such as `lead_slow`,
`signal_stop`, and `gap_available`, together with location and modality. The
seven categories provide the fixed Z1 capacity; the refinements provide
additional resolution for prioritization.

### Z2: typed support relation

Z2 describes how a claimed Z1 component supports one axis of the Z3 decision.
The current local roles are:

```text
proximate_cause
enabling_condition
blocking_condition
constraint
route_requirement
regulatory_requirement
```

This study operationalizes Z2 with these six typed roles. A Z3 decision lacking
a recognized supporting component receives an `unobserved` diagnostic clause,
which sits outside the six-role capacity.

### Z3: high-level decision

Z3 is the simultaneous Table 1 decision pair:

```text
(longitudinal decision, lateral decision)
```

The closed axes contain eight longitudinal values and nine lateral values,
including `none` on each axis. Excluding the empty `none|none` state gives

```text
8 × 9 - 1 = 71 non-empty Z3 decision pairs.
```

The number 71 is the **Z3 decision-pair capacity**. Complete-program coverage
additionally measures Z1 components, Z2 roles, typed paths, and continuations.

### Explicit continuations

A continuation is an explicitly stated subsequent reasoning stage. Each
continuation has one temporal guard:

```text
after_previous_complete
when_gap_available
when_path_clear
when_signal_permits
until_stop_or_yield_satisfied
```

The immediate stage uses `immediate`. Sequence coverage also records the
transition

```text
previous Z3 decision → continuation guard → next Z3 decision.
```

### Example

```text
“Keep distance from the slow lead vehicle, then change left when a gap opens.”
```

This produces two stages:

```text
Immediate stage
  Z1: slow lead vehicle — critical object
  Z2: proximate cause
  Z3: lead/obstacle following

Continuation
  guard: when gap available
  Z1: available gap — lane/lane-line condition
  Z2: enabling condition
  Z3: left lane change
```

Decision coverage counts the two decisions. Component–decision coverage counts
the critical-object/following and lane-condition/lane-change combinations.
Reasoning-path coverage additionally distinguishes the causal and enabling
relations. Continuation-aware coverage also counts the ordered transition
between the two stages.

## Coverage definitions

Reasoning coverage is evaluated at four levels:

| Coverage | What is counted |
| --- | --- |
| Decision coverage | The non-empty Z3 decision pair |
| Component–decision coverage | Every Z1 category paired with the Z3 decision |
| Reasoning-path coverage | Z1 components, Z2 relations, Z3 decisions, and their typed links |
| Continuation-aware coverage | Reasoning-path coverage for every explicit stage, plus transitions between stages |

The first three definitions use the immediate stage `S0`. The
continuation-aware definition adds only stages explicitly stated in the
reasoning text. Decision and component–decision coverage provide controlled
comparisons for measuring the contribution of Z2 relations and complete paths.

### Full typed-path items

Reasoning-path coverage includes items at several resolutions:

```text
Z1 category
Z1 component = category/predicate
Z1 detail = category/predicate/location/modality
Z2 role
Z3 longitudinal decision
Z3 lateral decision
Z3 decision pair
Z1 component → Z2 role
Z2 role → Z3 axis decision
Z1 component → Z2 role → Z3 axis decision
```

In this project, “Z1×Z2×Z3 coverage” refers to these realized typed paths, which
preserve the component–role–decision relationships found in each program.

## Denominators

### Fixed schema capacities

The report audits the closed and interpretable dimensions independently:

| Dimension | Capacity |
| --- | ---: |
| Table 2 Z1 categories | 7 |
| Local Z2 roles | 6 |
| Table 1 longitudinal decisions | 8 |
| Table 1 lateral decisions | 9 |
| Non-empty Z3 decision pairs | 71 |
| Z1 category × non-empty Z3 pair | 7 × 71 = 497 |

These dimensions are reported separately so each fraction retains a clear
semantic interpretation.

### Typed-path universe

A raw Z1-category × Z2-role × Z3-pair product contains 2,982 combinations.
Typed paths retain the feasible component–role–decision links realized by the
compiler, yielding a semantically grounded universe for prioritization.

The current typed-path denominator is consequently open and cohort-relative:

```text
D_k(E) = union of C_k(c) for every case c in fixed evaluation cohort E
```

where `C_k(c)` is the item set emitted by coverage definition `k` for case `c`.
This observed universe is used for greedy prioritization and coverage-growth
curves. It measures diversity among recognized outputs. The fixed schema
capacities above provide the corresponding externally defined audit dimensions.

A future externally fixed path denominator should be a feasibility-constrained
grammar reviewed independently of the evaluation outcomes. Until that exists,
reports must show both the fixed schema capacities and the observed item count.

### Ego-centric trajectory-grid control

Trajectory-grid coverage is a simple output-space comparison. It uses the
emitted trajectory in the ego rig frame, where `x` points forward and `y`
points left. The first 3 seconds of the trajectory are interpolated between
the recorded 100 ms waypoints and rasterized into origin-centered 2 m square
cells. Interpolation samples are at most 0.5 m apart. One case covers

```text
trajectory_grid:cell=(x index, y index)
```

for every occupied cell. Greedy prioritization uses the union of occupied cells
in the fixed cohort as its observed denominator.

This measures the spatial shape and extent of the policy output. It is not
global-map coverage: cell coordinates are reset around the ego vehicle for
every decision, so the same item can transfer across scenes and maps. It also
does not encode time, speed, acceleration, nearby agents, map topology, or the
reason for choosing a trajectory. Those omissions make it a useful naive
control rather than a complete behavioral abstraction.

## Test prioritization

Starting with an empty covered set, greedy prioritization repeatedly selects
the remaining case with maximum marginal gain:

```text
argmax_c |C_k(c) - already_covered_items|.
```

Equal-gain ties use a seeded random order. Once all observed items have been
covered, the remainder follows that seeded order.

Multi-round ordering (`--multi-round`) clears the covered set when no remaining
case adds coverage, then resumes greedy selection among the remaining cases.
Report the ordering rule and first-round saturation budget explicitly. See the
[previous experiment methodology](REASONING_COVERAGE_METHODOLOGY.md) for
multi-round comparisons and atomic-path ablations.

APFD evaluates how early outcomes appear in the coverage-produced order. The
large-cohort fault unit is one positive frame, so the random APFD baseline is
0.5 and a failure that persists across many 100 ms decisions contributes once
per frame. APFD integrates the whole order, so it can reward an arm that trails
random through every realistic review budget and catches up late; recall at a
fixed budget must therefore be reported alongside it and takes precedence when
the two disagree.

## Evaluation protocol

Every evaluation must report:

- the test unit: benchmark case, individual frame, or scene;
- the number of cases and positive frames;
- the exact valid/excluded cohort and overlap policy;
- fixed-capacity coverage and observed item counts;
- random, input, trajectory-summary, trajectory-grid, and text-bigram controls;
- mean APFD and randomized tie-break distribution;
- recall at fixed review budgets; and
- the precise reasoning-coverage definition being evaluated.

Coverage compilation and ordering are outcome-blind. Quantile-based input and
trajectory baselines are fit on the feature cohort independently of outcomes.

### Outcome units and dependence

Development benchmarks use independently labeled cases. Planned-safety studies
use emitted open-loop plans with recorded-agent motion and an ego-center
drivable-area test. Define valid labels and source-scene exclusions before
ordering, and exclude development-overlapping scenes from a source-disjoint
evaluation.

Adjacent frames within a scene are strongly dependent. Counting every positive
frame separately weights failures by duration; distinct-event claims require
an explicitly defined event unit. Report proxy scope and exclusions alongside
the evaluation.

## Implementation and reports

The closed Table vocabulary is in
[`alpamayo_tables.py`](../../src/tools/reasoning_coverage/alpamayo_tables.py). The
compiler and item construction are in
[`table_program.py`](../../src/tools/reasoning_coverage/table_program.py).
Trajectory-grid construction is in
[`trajectory_grid.py`](../../src/tools/reasoning_coverage/trajectory_grid.py).

Entry points are `reasoning_coverage.table_evaluation` and
`reasoning_coverage.planned_safety_evaluation`. Commands for reading archived
inputs and writing fresh output are in the
[archive index](../archive/reasoning_coverage/2026-09-06/README.md).

Detailed cohort counts, empirical comparisons, and original report paths are
preserved in the [historical definitions report](../archive/reasoning_coverage/2026-09-06/REASONING_COVERAGE_DEFINITIONS.md).

## Known limitations

- The deterministic compiler is not yet a human-validated semantic parser.
- Z2 roles and temporal guards are local research abstractions.
- Table 2 predicates, typed paths, and transitions have no external fixed
  denominator yet.
- The 150-case benchmark was repeatedly inspected and is exploratory.
- Report categorical-coverage saturation relative to the review budget.
- The trajectory grid is cohort-relative and mechanically related to
  trajectory-derived safety outcomes.
- Text-bigram controls are needed to distinguish structure from lexical diversity.
- The planned-safety outcomes are open-loop proxies rather than executed
  closed-loop safety outcomes.
