# Alpamayo reasoning-coverage pilot methodology

This document preserves the methodology of the initial retrospective pilot.
The [complete pilot report](../archive/reasoning_coverage/2026-09-06/REASONING_COVERAGE_PILOT.md)
contains its results, diagnostics, and historical proposed follow-up. These
methods document previous experiments; they do not lock the planned redesign.

## Research question

At a fixed review or regression-test budget, does coverage over a structured
reasoning state reveal more independently measured failures than random order,
scene-specification coverage, coarse input-image coverage, or trajectory
coverage?

This first experiment is retrospective. It can reject an unpromising coverage
definition and estimate an effect size, but it cannot establish generalization
or show that reasoning traces reduce the cost of producing model executions.

## Unit and outcomes

The unit is one `clip_id` in `benchmark.json`. Coverage construction never reads
the outcome fields. The primary outcome is `any_issue`, the union of:

- unreliable chain of thought;
- action-inconsistent chain of thought among traces labeled reliable; and
- unsafe planned trajectory.

The three components are also reported separately. These annotations are
independent of the prioritization algorithm, but their judge provenance may
still introduce correlated label error. A prospective experiment should use a
blinded human adjudication subset.

## Coverage arms

See [Deprecated pre-Table reasoning-coverage definitions](REASONING_COVERAGE_DEFINITIONS_DEPRECATED.md)
for the decomposition, denominator, and selection rule used by the original
benchmark and OOD arms.

| Arm | Coverage items |
| --- | --- |
| `random` | Random permutation; 200 seeds by default |
| `specification` | Unary and cross-field pairs from existing NuRec scene labels |
| `visual_stats` | Quantized global luminance, entropy, edge, and color statistics |
| `image_grid` | Quantized RGB values on an 8 by 8 image grid |
| `trajectory` | Quantized trajectory summaries and dominant action categories |
| `reasoning_node` | Longitudinal decision, lateral decision, and causal categories |
| `reasoning_graph` | Nodes plus decision pairs and cause-to-decision edges |
| `reasoning_fine` | Graph items plus open-vocabulary tags and atomic actions |
| `hybrid` | Specification, trajectory, and reasoning-graph items |

Continuous inputs are fit into corpus quantile bins. Every non-random arm uses
the same greedy rule: repeatedly select the case that adds the most uncovered
items, with randomized equal-gain ties. `image_grid` is a reproducible coarse
input-space baseline, not full-resolution pixel coverage or neural activation
coverage.

The primary reasoning arm should be `reasoning_graph`. `reasoning_node` is an
ablation. `reasoning_fine` is exploratory because open-vocabulary tags have a
large singleton space and are sensitive to annotator wording.

## Metrics

The primary metric is APFD, treating each failing case as one failure to be
found. Recall at 10%, 25%, and 50% of the case budget is also reported. The
reported intervals describe randomized tie-breaking only; they are not
confidence intervals over independently sampled scenes.

Runtime, labeling time, model-token cost, and artifact-storage cost must be
recorded in the prospective phase. Without those measurements, this pilot tests
failure-discovery effectiveness rather than end-to-end efficiency.

## Label-audit procedure

Review reasoning text with images, trajectories, and failure outcomes hidden.
The coarse audit sampled one case per observed decision pair. The program audit
sampled unique texts across single/multi-step and resolved-decision strata.
Balanced stress samples diagnose parser errors rather than estimate population
accuracy. Codex-assisted labels require researcher review and are not ground
truth. Audit records are in the [archive](../archive/reasoning_coverage/2026-09-06/README.md).

## Per-plan safety proxy

The OOD ASLs also support an exhaustive outcome at each policy decision rather
than one outcome for a complete scene trace. At every decision, the labeler:

1. uses the 65 poses emitted by Alpamayo at their original 100 ms sampling;
2. applies the rollout's rig-to-AABB-center transform rather than treating the
   rig origin as the vehicle center;
3. checks strictly future poses through 3 seconds against the union of
   drivable-lane polygons; and
4. joins AlpaSim's existing `open_loop_collision` result, which intersects the
   planned ego box with recorded agents at matching future timestamps.

The explicit outcome names are
`planned_ego_center_outside_drivable_area_3s_100ms`,
`open_loop_collision_with_recorded_agents_3s_100ms`, and their union
`open_loop_safety_violation_3s_100ms`. They describe the hypothetical emitted
plan, not executed motion. No path is interpolated between emitted waypoints,
and recorded agents do not react to the counterfactual ego plan.

Exclude plans whose current vehicle center is already outside the drivable
union or whose future plan leaves the available map extent. Remove source-scene
overlap with development data before evaluating the common valid cohort.

The pilot also evaluated episodes: maximal runs of positive frames exactly
100 ms apart within one source scene, detected when any member is selected.
Random episode APFD can exceed 0.5 because an episode has multiple opportunities
for discovery. Later studies used individual positive frames; comparisons must
state the fault unit.

## Inferential uncertainty

The OOD analysis used paired tie-seed deltas, outcome-permutation tests, and a
source-scene percentile bootstrap. Its bootstrap held fitted abstractions and
orders fixed. Tie-seed intervals measure ordering variation, not uncertainty
across new scenes. A confirmatory source bootstrap should refit bins and
orderings within each resample. No inferential test was used for sparse
positive-unlabeled manual outcomes.

## Fixed causal-obligation denominator prototype

The earlier program arms define their item universe from model outputs observed
in the evaluated cohort. That is useful for prioritization, but it makes the
denominator self-referential: a behavior the model never emits cannot be an
uncovered item. The `alpamayo-cro-v1` prototype therefore freezes an
outcome-blind denominator before reading a trace or safety label.

The intended unit is a causal reasoning obligation (CRO): an independently
applicable situation, its critical component and relation to ego, the driving
objective, and a constrained response class. Version 1 implements only the
part supported by current ASL state. It reads the exact rollout logs recorded
in the planned-safety artifact and uses actor identity, class, AABB, pose, and
finite-difference velocity. It does not read the planned-safety outcomes while
triggering obligations.

The closed v1 catalog has 27 exact variants:

- 18 nearest-lead variants: light/heavy vehicle, stopped/slower/not-slower,
  and near/mid/far;
- 9 transverse-conflict variants currently named `crossing_agent_yield`:
  light/heavy vehicle or VRU and near/mid/far; and
- seven top-level critical-component domains, of which only
  `critical_objects` is independently observable in v1. Signals, controls,
  road events, lanes, routing, and ODD constraints remain explicitly
  uninstrumented rather than being inferred from Alpamayo's text.

Each applicable obligation is joined to four trace-discharge states and to the
71 non-empty pairs in Alpamayo's published high-level decision space (eight
longitudinal choices including none by nine lateral choices including none,
excluding none/none). The nested coverage arms are applicability, discharge
status, and conditioned decision. Report frozen capacity separately from observed cells. Geometric
conflict alone does not establish right-of-way or an obligation to yield; the
`crossing_agent_yield` proxy must not be treated as semantic ground truth.

## Reproduction and prospective design

Use the [pre-Table definitions](REASONING_COVERAGE_DEFINITIONS_DEPRECATED.md)
for program arms and the [archive index](../archive/reasoning_coverage/2026-09-06/README.md)
for historical commands and relocated inputs.

For a new evaluation, freeze the schema, compiler, item weighting, quantile
procedure, and selector before acquiring holdout outcomes. Split by source
scene and lock inclusion rules, primary outcome, comparator, minimum useful
improvement, and parser-audit gates. Produce and hash all coverage orderings
while outcomes are inaccessible. Treat exploratory ablations separately.

An online efficiency study must derive coverage from previews available before
execution, execute candidate scenarios in batches, and measure failures found
per GPU-hour including reasoning-generation and label-extraction costs.
