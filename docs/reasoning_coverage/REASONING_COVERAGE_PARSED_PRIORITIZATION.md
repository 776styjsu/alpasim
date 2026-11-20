# Prioritization with parsed reasoning paths

This experiment evaluates the new decision–role–proposition representation on
the large planned-safety cohort. The vocabulary is frozen in
[`decision_role_proposition_v1.json`](../../data/reasoning_coverage/decision_role_proposition_v1.json).
The parser outputs are inputs to this evaluation; the evaluator does not call
an LLM or modify the parses.

## Coverage definitions

| Arm | Item definition | Unpruned capacity |
| --- | --- | ---: |
| Independent | Complete decision variant, role, or family-qualified proposition field-value, counted separately | 151 |
| Fully additive | Decision modifiers counted independently within their decision family; each modifier-free decision family contributes one item; roles and proposition field-values are separate | 146 |
| Single-field paths | Complete decision variant × explicit role × one proposition field-value | 17,888 |
| Paths ≤2 fields | Complete decision variant × explicit role × one or two fields from one proposition instance | 124,872 |
| All partial paths | Complete decision variant × explicit role × every nonempty subset of fields from one proposition instance | 2,764,556 |
| Z3 values only | One family-qualified proposition field-value, with no decision or role axis | 104 |
| Z3 propositions only | Every nonempty subset of fields from one proposition instance, with no decision or role axis | 16,073 |

Keys include family and field names, so identically spelled values in different
fields remain distinct. An unstated value earns no credit. Explicit Boolean
false remains a substantive value. A missing required decision modifier prevents
complete-variant and path credit, but stated modifiers still contribute to the
fully additive arm. An absent commitment on the other control axis has no effect.

Independent arms count unlinked mentions. Path arms require a supplied relation;
co-occurrence never creates an edge. Fields are never combined across proposition
instances. All parsed decision statuses, temporal scopes, and proposition
assertion modes are retained, including blocked candidates and conditional
statements. Metadata, IDs, and evidence are not coverage dimensions. These arms
measure expressed reasoning diversity, not the truth of scene propositions.

Fixed capacities describe the schema. They do not change greedy rankings:
dividing every marginal gain within an arm by the same constant preserves its
ordering. The evaluator reports both fixed capacities and observed item counts.

## Cohort and controls

Frames come from the archived `ood_hazard_2604.features.json`. Exclude source
scenes represented in the development `benchmark.json`, then keep frames marked
valid by the archived `ood_hazard_2604.planned_safety_3s_100ms.json`. Eligibility
does not depend on whether a valid label is positive or negative.

The resulting cohort contains 14,867 frames from 91 scenes, with 2,371 unique
reasoning strings. Unwrap the stored reasoning with the existing text-normalization
helper and match exactly to `experiments/cot_parse/batches/`. Join parsed outputs
by `trace_id`. Missing parses, duplicate identifiers, unsupported enum values,
and missing trajectory data cause an error. Frames whose valid parses produce
empty coverage remain in every arm's ordering.

The trajectory-grid control uses the existing rasterizer: first 3 seconds of
the emitted ego-rig-frame plan, origin-centered 2 m cells, and interpolation
samples no more than 0.5 m apart. The denominator is the union of observed cells.
A seeded random permutation provides an additional baseline.

## Ordering and outcomes

Use 50 repeats with seeds 42–91, pairing the same seed across all methods.
Cases are initially sorted by frame ID. Each coverage arm selects maximum
marginal item gain with seeded tie-breaking.

- Multi-round greedy resets the covered set when remaining cases add no new
  coverage, then resumes selection. This is the main comparison.
- Single-round greedy appends the seeded random tail after saturation. This is
  a sensitivity check.
- In both modes, a remainder consisting entirely of empty item sets follows
  the seeded order.

Persist every ordering before computing outcome metrics. Report collision,
ego-center-off-road, and their union separately. One positive frame is one fault
unit; adjacent positive frames are not merged into episodes.

Report APFD, recall and detected-positive counts at 1%, 5%, 10%, and 25% budgets,
using `ceil(fraction × cohort_size)` reviewed frames. Report paired APFD and
detection-count differences from trajectory grid. Intervals describe the
2.5–97.5 percentiles across tie seeds, not confidence intervals over new scenes.

For each reasoning method and budget, compare its detected fault set with the
grid's set from the same repeat. Report reasoning-only, grid-only, and shared
positives. A frame is robustly exclusive to reasoning when its detection
probability is at least 0.75 for reasoning and at most 0.25 for grid; also report
the reverse. Exclusivity describes this comparison at this budget, not a fault
that other techniques cannot discover. Frame counts weight persistent failures
by duration and do not estimate the number of distinct physical incidents.

## Findings

**No fault-detection advantage for reasoning coverage is established by this
run.** Coverage-guided selection of any kind beats random ordering on the
off-road outcome, but reasoning coverage is not separable from the
trajectory-grid control, and on collisions it is not separable from random.

Two interval types must be kept apart. Intervals over tie-breaking seeds say
whether two orderings differ on this fixed cohort. They are not generalization
claims. Frames within a scene are strongly dependent, so a cluster bootstrap
over the 91 scenes is the appropriate check, and it is much wider. Only the
second kind supports a claim about new scenes.

### Against the random baseline

The grid is a weak reference on collisions: at a 10% budget it detects 72.4
against random's 140.7, so it is substantially *worse* than random there.
Comparisons stated only against the grid therefore overstate the reasoning
arms. Paired against random over seeds, detected positives:

| Budget | Collision (paths ≤2) | Off-road (paths ≤2) | Union (paths ≤2) |
| --- | ---: | ---: | ---: |
| 1% | −0.9 | +42.8 * | +36.1 * |
| 5% | +3.4 | +191.3 * | +162.4 * |
| 10% | +11.9 | +317.3 * | +268.9 * |
| 25% | −27.1 * | +305.8 * | +210.2 * |

`*` marks intervals excluding zero over seeds. On collisions no reasoning arm
separates from random at 1%, 5% or 10%, and at 25% every arm is significantly
worse. The off-road and union gains are large, and they survive the scene-level
cluster bootstrap (paths ≤2 at 10%: +314 [+199, +425] off-road, +265 [+148,
+374] union).

### Against the trajectory grid

The apparent small-budget win over the grid does not survive scene resampling.
Paired paths ≤2 minus grid, cluster bootstrap over scenes:

| Budget | Collision | Off-road | Union |
| --- | ---: | ---: | ---: |
| 5% | +38.9 [−12, +94] | +3.0 [−95, +100] | +10.2 [−91, +109] |
| 10% | +74.5 [−22, +170] | +42.6 [−102, +173] | +66.8 [−87, +207] |
| 25% | +40.2 [−181, +236] | −176.6 [−360, +2] | −155.6 [−434, +81] |

Every interval includes zero. With 91 scenes the comparison cannot resolve a
difference of this size in either direction; this is an inconclusive result, not
a demonstrated equivalence.

The grid also holds a structural edge that the headline APFD table hides.
Ego-centre off-road is a geometric function of the same planned trajectory the
grid rasterises, so grid coverage approximates that outcome's mechanism
directly. Collision is the less circular comparison, and it is the one where
neither method beats random.

### Where the orderings diverge

Over the full ordering the grid attains the higher APFD on every outcome
(union 0.6217 against 0.5223 for paths ≤2). The union lead of the path arms is
lost after roughly 16.5% of the cohort, the collision lead at 28%. Two effects
combine: the grid ordering reaches a collision-dense block between 25% and
32.5% of budget, where its marginal rate is 17–22% against a 9.32% base rate,
while over the same stretch the path arms fall to 4.8–6.8%, below base rate.
Only 581 distinct item sets exist across 10,731 frames, so after the first
greedy rounds the remaining selections are near-duplicates. Item-set diversity,
not denominator size, is the binding constraint.

### Dropping the decision and role axes (Z3 ablation)

Scoring propositions alone removes the requirement for a complete decision
variant and a role link, which cuts the deferred tail from 4,136 frames to
1,575. It does not help. Union APFD is unchanged (0.5226 for Z3 values against
0.5223 for paths ≤2), and detection is worse at every budget: against random at
10%, +98.8 for Z3 values against +268.9 for paths ≤2. On collisions the Z3 arms
are below random at all four budgets.

The tail improvement is cancelled by a loss of resolution. Distinct item sets
fall from 581 to 417, because reasoning that differs in what it justifies
collapses into the same proposition bag: a stopped vehicle that motivates
braking and one that blocks a lane change become indistinguishable. Item-set
diversity, not denominator size or tail placement, is what drives these
orderings — enlarging the space from 124,872 to 2,764,556 changed nothing,
while removing the 43 × 4 decision-role context cost roughly two thirds of the
gain over random. This supports keeping decision × role × proposition as the
primary unit.

### Frames with no extractable coverage are fault-enriched

Frames whose valid parses yield no items are more likely to fail, so every arm
sorts its densest material into the tail:

| Arm | Empty frames | Collision rate | Off-road rate | Union rate |
| --- | ---: | ---: | ---: | ---: |
| Independent | 648 | 16.36% | 25.15% | 36.42% |
| Paths ≤2 fields | 4,136 | 9.72% | 13.85% | 21.57% |
| Non-empty (paths ≤2) | 10,731 | 9.17% | 9.10% | 16.96% |
| Cohort | 14,867 | 9.32% | 10.43% | 18.24% |

For the independent arm the union rate is twice the cohort rate. Absent
extractable reasoning is a stronger fault signal than any single coverage item,
and the current protocol spends it last. Whether this reflects degenerate model
output or parser failure is unresolved and needs the pending human accuracy
audit; the two have different remedies, so the effect should not be built into
a selection rule before that audit.

Diagnostics derive from committed artifacts:
`experiments/scratch/prioritization_posthoc_20260908.py` writes
`posthoc_diagnostics.json`, and
`experiments/scratch/prioritization_baselines_20260908.py` writes
`baseline_robustness.json`, both into the results directory.

## Reproduction and artifacts

From the repository root:

```bash
PYTHONPATH=src/tools:src/utils uv run --no-sync python -m reasoning_coverage.parsed_evaluation \
  --output-dir experiments/results/reasoning_coverage_prioritization_rerun \
  --repeats 50 --seed 42 --workers 8
```

The output directory must be new. The evaluator saves input and implementation
hashes, per-trajectory hashes, per-frame coverage sets, complete order arrays,
cohort membership, APFD and budget tables, per-fault detection probabilities,
and PNG/PDF figures.

The [September 8 results](../../experiments/results/reasoning_coverage_prioritization_2026-09-08/REPORT.md)
contain the completed comparisons. Generated data remain under `experiments/`.
The source-disjoint cohort was used in previous exploratory work; this is a
retrospective comparison, not an untouched holdout. Parser validity and
open-loop proxy validity remain separate from coverage and prioritization.
