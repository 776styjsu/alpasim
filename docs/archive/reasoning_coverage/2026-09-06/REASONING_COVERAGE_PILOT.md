# Alpamayo reasoning-coverage pilot

> [!WARNING]
> This report documents the initial pilot. Current experiments use the
> [reasoning-coverage definitions](REASONING_COVERAGE_DEFINITIONS.md). See the
> [cross-cohort experiment analysis](REASONING_COVERAGE_EXPERIMENT_ANALYSIS.md)
> for fault-detection curves and failure characterization.

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

See [Deprecated pre-Table reasoning-coverage definitions](../../../REASONING_COVERAGE_DEFINITIONS_DEPRECATED.md)
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

## Label audit

[`codex_audit_v1.json`](../../../../experiments/archive/reasoning_coverage/2026-09-06/data/reasoning_coverage/codex_audit_v1.json) contains
one text-only review from every decision pair observed before audit (21 cases).
The annotator did not use images, trajectories, or failure outcomes. These are
Codex-assisted research labels, explicitly not ground truth, and require
researcher review.

The audit is a coarse-label sensitivity check. It is not valid evidence for the
fine-grained arm because only audited cases use the revised tag vocabulary.

## Run

From the repository root:

```bash
PYTHONPATH=src/tools uv run python -m reasoning_coverage \
  --benchmark benchmark.json \
  --output runs/reasoning_coverage/benchmark_pilot.heuristic.json \
  --curve-csv runs/reasoning_coverage/benchmark_pilot.heuristic.curves.csv

PYTHONPATH=src/tools uv run python -m reasoning_coverage \
  --benchmark benchmark.json \
  --audit-labels data/reasoning_coverage/codex_audit_v1.json \
  --output runs/reasoning_coverage/benchmark_pilot.audit.json \
  --curve-csv runs/reasoning_coverage/benchmark_pilot.audit.curves.csv
```

The JSON artifact records configuration, artifact availability, label sources,
quantization edges, representative orders, coverage curves, failure-recall
curves, and APFD summaries.

The closed-program follow-up uses the same IID command with explicit artifact
names, plus two OOD evaluators:

```bash
PYTHONPATH=src/tools uv run python -m reasoning_coverage \
  --benchmark benchmark.json \
  --output runs/reasoning_coverage/benchmark_pilot.program_v1.json \
  --curve-csv runs/reasoning_coverage/benchmark_pilot.program_v1.curves.csv \
  --repeats 200 --seed 42

PYTHONPATH=src/tools uv run python -m reasoning_coverage.ood_evaluation \
  --output runs/reasoning_coverage/ood_hazard_2604.program_v3.json \
  --scene-repeats 200 --frame-repeats 50 \
  --permutations 10000 --bootstraps 10000 --seed 42 --workers 8

PYTHONPATH=src/tools uv run python -m reasoning_coverage.temporal_sensitivity \
  --output runs/reasoning_coverage/ood_hazard_2604.prefix25.json \
  --prefix-fraction 0.25 --repeats 200 \
  --permutations 10000 --bootstraps 10000 --seed 42 --workers 8

PYTHONPATH=src/tools uv run python -m trajectory_safety.planned_center_labels \
  --output runs/reasoning_coverage/ood_hazard_2604.planned_safety_3s_100ms.json \
  --horizon-s 3.0 --workers 8

PYTHONPATH=src/tools uv run python -m reasoning_coverage.planned_safety_evaluation \
  --labels runs/reasoning_coverage/ood_hazard_2604.planned_safety_3s_100ms.json \
  --output runs/reasoning_coverage/ood_hazard_2604.planned_safety_coverage.json \
  --repeats 50 --seed 42 --workers 8
```

The follow-up artifacts are
[`benchmark_pilot.program_v1.json`](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/benchmark_pilot.program_v1.json),
[`benchmark_pilot.program_v1_support_ablation.json`](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/benchmark_pilot.program_v1_support_ablation.json),
[`ood_hazard_2604.program_v3.json`](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/ood_hazard_2604.program_v3.json),
[`ood_hazard_2604.prefix25.json`](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/ood_hazard_2604.prefix25.json),
[`ood_hazard_2604.planned_safety_3s_100ms.json`](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/ood_hazard_2604.planned_safety_3s_100ms.json),
[`ood_hazard_2604.planned_safety_coverage.json`](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/ood_hazard_2604.planned_safety_coverage.json),
and
[`reasoning_program_v1_ood_audit.json`](../../../../experiments/archive/reasoning_coverage/2026-09-06/data/reasoning_coverage/reasoning_program_v1_ood_audit.json).
The support-ablation, OOD, prefix, and audit artifacts record input and
implementation hashes; the main IID report records paths and configuration but
does not record hashes. The 79 MB feature cache is reusable but is not itself
an evaluation result. The IID support ablation is a recorded post-hoc
sensitivity artifact; there is currently no dedicated CLI for regenerating it.

## Development result (2026-08-21)

All 150 cases had trajectory summaries and images. The benchmark contains 50
unreliable traces, 26 action-inconsistent traces among those labeled reliable,
17 unsafe trajectories, and 78 cases with at least one issue.

| Arm | APFD, any issue | Recall at 25% budget |
| --- | ---: | ---: |
| Random | 0.500 | 0.251 |
| Specification | 0.499 | 0.250 |
| Image grid | 0.499 | 0.251 |
| Trajectory | 0.513 | 0.279 |
| Reasoning graph, original labels | 0.520 | 0.293 |
| Reasoning graph, 21 audited labels | 0.518 | 0.290 |
| Hybrid, 21 audited labels | 0.531 | 0.300 |

The graph result is stable to the limited audit, but its improvement is small:
0.018 APFD over random and 0.005 over trajectory after audit. At 25% budget it
finds about three more of the 78 issue cases than random, and fewer than one
more than trajectory. It does not meet the proposed 0.05 confirmatory effect
threshold.

The gain is also outcome-specific. With audited labels, reasoning-graph APFD is
0.539 for unreliable traces, 0.483 for action inconsistency, and 0.511 for
unsafe trajectories, compared with random means of 0.499, 0.503, and 0.496.
The current signal therefore concerns reasoning reliability more than behavioral
failure discovery.

The audited fine-grained arm reaches 0.558 APFD, but this is not interpretable:
19 of the 21 audited cases changed open-vocabulary tags, creating a mixed
annotation vocabulary and extra singleton coverage items. A full relabel or a
closed tag ontology is required before comparing the fine-grained arm.

## Closed reasoning-program follow-up (2026-08-21)

This section reports an adaptive exploratory follow-up. The OOD suite, unsafe
examples, and outcomes were inspected while the analysis was being developed.
The compiler and ablations were also revised in response to diagnostics. These
results are useful for deciding what to lock next, but they are not a
confirmatory holdout result.

### `reasoning_program_v1`

[`program.py`](../../../../src/tools/reasoning_coverage/program.py) is a deterministic,
text-only compiler. It maps each chain-of-thought string to a closed
representation with facts (predicate, location, and modality), objectives,
longitudinal and lateral decisions, longitudinal and lateral actions, entry
conditions, and ordered steps. It never reads an image, trajectory, result
summary, or unsafe-event label. Missing or ambiguous decisions and actions do
not earn primary coverage credit. All reported follow-up artifacts use compiler
SHA-256 `30ebe1ea8cd880127ad93b1102289dd88aa41eb736815206c015accb207dcd6b`.

The program arms are cumulative:

| Arm | Added coverage items |
| --- | --- |
| `program_node` | Fact, objective, decision, action, and entry-condition nodes |
| `program_edge` | Nodes plus allowed fact-to-objective, objective-to-decision, decision-to-action, and joint decision/action edges |
| `program_path` | Edges plus within-step causal paths and within-trace temporal paths |
| `program_trace` | Path items plus transitions between consecutive frame-level programs |

`text_bigram` is the required non-semantic control: normalized word bigrams
from exactly the same reasoning text. Greedy ordering and randomized tie
handling are otherwise identical. `program_hybrid` on IID combines
specification, trajectory, and program-path coverage; the OOD `hybrid` combines
trajectory and program-trace coverage.

### IID result on the 150-case development benchmark

The values below are mean APFD over the same 200 predetermined tie-breaking
seeds. Six decimal places are retained from the artifacts; they should not be
read as six digits of statistical precision.

| Arm | Any issue (78) | Unreliable CoT (50) | Inconsistent reliable CoT (26) | Unsafe trajectory (17) |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.500374 | 0.499438 | 0.502778 | 0.496027 |
| Specification | 0.498797 | 0.496708 | 0.510826 | 0.499786 |
| Visual statistics | 0.500504 | 0.499545 | 0.502594 | 0.496155 |
| Image grid | 0.499200 | 0.499597 | 0.498622 | 0.491890 |
| Trajectory | 0.512835 | 0.501084 | 0.539535 | 0.493812 |
| Legacy reasoning graph | 0.519553 | 0.536939 | 0.493027 | 0.529143 |
| Text bigram | 0.575200 | 0.643296 | 0.468282 | 0.496424 |
| Program node | 0.547376 | 0.582719 | 0.492669 | 0.534761 |
| Program edge | 0.556623 | 0.602030 | 0.483840 | 0.542388 |
| Program path | 0.558172 | 0.607571 | 0.478581 | 0.556422 |
| Program hybrid | 0.549220 | 0.582688 | 0.506832 | 0.563382 |

The structured arms improve `any_issue` APFD by 0.047--0.058 over random, but
the raw text-bigram arm is higher than every structured arm. The apparent gain
is concentrated in unreliable-CoT labels and does not extend to action
inconsistency. This does not establish that semantic structure is responsible
for the ordering.

### IID item-support sensitivity

Item support is the number of distinct cases containing an item. Support 1 is
the raw arm; support 2 or 5 removes rarer items before ordering, without using
outcomes. This was a post-hoc analysis.

| Arm | Items, support >= 1 | APFD | Items, support >= 2 | APFD | Items, support >= 5 | APFD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text bigram | 959 | 0.575200 | 318 | 0.524638 | 121 | 0.506906 |
| Program node | 184 | 0.547376 | 127 | 0.530415 | 69 | 0.509958 |
| Program edge | 279 | 0.556623 | 196 | 0.531908 | 106 | 0.518608 |
| Program path | 356 | 0.558172 | 216 | 0.524465 | 107 | 0.518919 |

Random is 0.500374 in the same 200 seeds. Removing rare items substantially
reduces every gain. Rare-state prioritization can be a legitimate property of
coverage, but here it is also consistent with wording, parser, or
dataset-selection artifacts. The ablation therefore weakens a semantic
interpretation of the raw result.

### OOD cohort and provenance

The OOD run began with 18,564 frame records from 102 scenes with extracted
reasoning traces. Nine source scenes overlapped the development benchmark and
were excluded before evaluation. Of the remaining 93 scenes, 91 had complete
closed-loop outcome summaries, yielding a source-disjoint evaluation cohort of
91 scenes and 16,562 frames (182 frames per scene).

The separately supplied file contained 20 known unsafe frame labels. Three
belonged to development-overlapping source scenes, leaving 17 strict known
unsafe frames; one of those scenes lacked a complete outcome summary. The
91-scene cohort therefore contains 16 manually flagged unsafe scenes. Its
outcomes are:

- 22 exhaustive closed-loop failures (`passed == false`);
- 38 exhaustive closed-loop safety events, defined as a positive
  `collision_at_fault`, `offroad`, `wrong_lane`, or
  `safety_monitor_triggered` metric; and
- 16 manually flagged unsafe scenes, which are positive-unlabeled rather than
  exhaustive outcomes.

Manual plan unsafety and closed-loop failure are not interchangeable. An
unflagged scene is not known safe, and a manually unsafe planned timestep need
not cause the closed-loop rollout to fail. Evaluating the 20 positive examples
alone would be uninformative: if every case is positive, APFD is 0.5 for every
ordering. The frame challenge instead embeds the retained positives in the
larger trace pool, while preserving their positive-unlabeled status. The OOD
hazard suite is also a targeted corpus rather than a population sample.

### OOD full-trace scene prioritization

Coverage was unioned across each scene's complete 182-frame trace. These are
mean APFD values over 200 tie-breaking seeds.

| Arm | Closed-loop failure (22) | Manual unsafe scene (16, PUL) | Closed-loop safety event (38) |
| --- | ---: | ---: | ---: |
| Random | 0.502040 | 0.498328 | 0.503655 |
| Visual statistics | 0.518554 | 0.493527 | 0.511316 |
| Image grid | 0.513217 | 0.487720 | 0.505591 |
| Trajectory | 0.478581 | 0.466707 | 0.503133 |
| Text bigram | 0.579528 | 0.552888 | 0.572379 |
| Program node | 0.549383 | 0.500903 | 0.564475 |
| Program edge | 0.565887 | 0.536247 | 0.565623 |
| Program path | 0.551081 | 0.559657 | 0.541667 |
| Program trace | 0.544168 | 0.540388 | 0.547918 |
| Hybrid | 0.537240 | 0.539749 | 0.548778 |

Program edge is 0.063846 above random for closed-loop failure and 0.061968
above random for the safety-event outcome. Raw text bigrams are higher still:
0.077488 and 0.068723 above random. Frequency filtering again changes the
picture: for failure, text bigram drops from 0.579528 to 0.544810 at support 2,
program edge drops from 0.565887 to 0.491908, and program trace changes from
0.544168 to 0.477517 at support 2 and 0.522318 at support 5.

At the frame level, the 17 supplied unsafe frames among 16,926 strict frames
form only a positive-unlabeled retrieval challenge. APFD is near random:
random 0.502924, image grid 0.504139, text bigram 0.499382, program edge
0.499667, program path 0.498379, and program trace 0.496006. Because the other
16,909 frames were not exhaustively adjudicated, these numbers must not be
interpreted as unsafe-versus-safe classification performance.

### Inferential uncertainty

For exhaustive OOD outcomes, the artifact reports a paired delta distribution
over tie-breaking seeds, an outcome-permutation test, and a source-scene
percentile bootstrap. The bootstrap resamples scenes but holds the fitted
coverage abstractions and orderings fixed.

| Outcome | Arm vs random | Mean delta | Tie-seed 2.5--97.5% | Two-sided permutation p | Source-bootstrap 95% interval |
| --- | --- | ---: | ---: | ---: | ---: |
| Closed-loop failure | Text bigram | 0.077488 | [-0.023526, 0.175899] | 0.149385 | [-0.055721, 0.203530] |
| Closed-loop failure | Program edge | 0.063846 | [-0.012050, 0.142420] | 0.175682 | [-0.047881, 0.175923] |
| Safety event | Text bigram | 0.068723 | [0.010244, 0.128499] | 0.052095 | [-0.023784, 0.160365] |
| Safety event | Program edge | 0.061968 | [0.014438, 0.108213] | 0.051695 | [-0.020110, 0.142232] |

Every source-bootstrap interval in this core table includes zero. The
permutation values are Monte Carlo estimates from 10,000 permutations and the
bootstrap values use 10,000 resamples. More importantly, the analysis examined
many arms, three outcomes, multiple support thresholds, and a temporal
sensitivity after inspecting the data; no multiplicity correction was applied.
The p-values are diagnostics, not confirmatory evidence. The 200-seed intervals
measure ordering instability from tied coverage gains, not uncertainty over a
new scene population. No inferential test is reported for the positive-unlabeled
manual labels.

### Full trace versus a 25% prefix

The full-trace ordering can use states observed after the event it is supposed
to prioritize. A temporal sensitivity reran coverage on the first 46 of 182
frames per scene. The prefix is equal length, but is not guaranteed to be
strictly pre-event.

| Arm | Failure full | Failure prefix | Manual PUL full | Manual PUL prefix | Safety full | Safety prefix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.502040 | 0.502040 | 0.498328 | 0.498328 | 0.503655 | 0.503655 |
| Visual statistics | 0.518554 | 0.501159 | 0.493527 | 0.500367 | 0.511316 | 0.502724 |
| Image grid | 0.513217 | 0.504850 | 0.487720 | 0.509018 | 0.505591 | 0.524430 |
| Trajectory | 0.478581 | 0.509326 | 0.466707 | 0.481484 | 0.503133 | 0.512413 |
| Text bigram | 0.579528 | 0.474283 | 0.552888 | 0.499890 | 0.572379 | 0.488944 |
| Program node | 0.549383 | 0.487118 | 0.500903 | 0.444028 | 0.564475 | 0.494741 |
| Program edge | 0.565887 | 0.445302 | 0.536247 | 0.485199 | 0.565623 | 0.476395 |
| Program path | 0.551081 | 0.462925 | 0.559657 | 0.496054 | 0.541667 | 0.471472 |
| Program trace | 0.544168 | 0.425415 | 0.540388 | 0.481302 | 0.547918 | 0.458524 |
| Hybrid | 0.537240 | 0.430554 | 0.539749 | 0.474179 | 0.548778 | 0.462016 |

All 22 failure-positive, 16 manually flagged, and 38 safety-event-positive
full traces contain post-event frames. Even the 25% prefix contains post-event
frames for 7/22 failures, 6/16 manual events, and 15/38 safety events. A common
strictly pre-event horizon collapses to zero frames for the two exhaustive
outcomes because some events are already present at or before the first trace
frame; it is 13 frames for the manual PUL outcome. The structured and lexical
full-trace gains disappear or reverse in the prefix. This is strong evidence
that the full-trace result is temporally confounded, not evidence that the
reasoning abstraction anticipates hazards.

### Outcome-blinded parser audit

The compiler was audited on 30 unique OOD chain-of-thought strings sampled with
seed 2604 and balanced across single/multi-step outputs and resolved-decision
strata. The auditor saw only the text and compiled representation--not images,
trajectories, scene results, or unsafe labels. This stress sample deliberately
overrepresents rare parser strata, so the rates are diagnostics rather than
population accuracy estimates. `Exact` means no obvious false positive or
false negative was found for that dimension.

| Dimension | Exact / 30 | Samples with false positives | Samples with false negatives |
| --- | ---: | ---: | ---: |
| Facts | 8 | 9 | 18 |
| Objectives | 10 | 6 | 19 |
| Longitudinal decision | 19 | 0 | 11 |
| Lateral decision | 17 | 3 | 11 |
| Actions | 11 | 3 | 17 |
| Segmentation | 22 | 1 | 8 |

At least seven samples also had an obvious fact-detail scope error. The audit
shows that `reasoning_program_v1` is a reproducible abstraction, but not yet a
validated semantic labeler. Its APFD may be driven partly by deterministic
lexical and parser artifacts.

### Prospective per-plan safety proxy

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

All 18,564 extracted decisions were labeled. After excluding nine development-
overlapping source scenes, 16,926 decisions from 93 scenes remained. The common
valid cohort contains 14,867 decisions from 91 scenes. It excludes 1,945
decisions whose current AABB center was already outside the drivable union and
114 whose future plan left the rectangular extent of the available map. The
exhaustive outcomes in that common cohort are:

| Outcome | Positive frames | Consecutive 100 ms episodes |
| --- | ---: | ---: |
| Recorded-agent collision | 1,386 | 120 |
| Center off-road | 1,550 | 249 |
| Collision or center off-road | 2,712 | 318 |

An episode is a maximal run of positive frames exactly 100 ms apart within one
source scene. Episode APFD treats an episode as detected when its first member
is selected; this is primary because adjacent positive frames are not
independent. Random episode APFD is greater than 0.5 because a multi-frame
episode has several opportunities for early detection, so only within-outcome
deltas between arms are meaningful. These are means over 50 outcome-blind
tie-breaking seeds:

| Arm | Collision episode APFD | Center-off-road episode APFD | Union episode APFD |
| --- | ---: | ---: | ---: |
| Random | 0.656516 | 0.663268 | 0.669470 |
| Visual statistics | 0.656506 | 0.663267 | 0.669467 |
| Image grid | 0.656442 | 0.663302 | 0.669484 |
| Trajectory | 0.656335 | 0.664012 | 0.669999 |
| Text bigram | 0.695149 | 0.729138 | 0.718353 |
| Text bigram, support >= 2 | 0.666363 | 0.694461 | 0.690442 |
| Program node | 0.661479 | 0.678653 | 0.680582 |
| Program edge | 0.661005 | 0.680895 | 0.682258 |
| Program path | 0.666907 | 0.683879 | 0.685897 |
| Program trace | 0.665859 | 0.686276 | 0.687020 |
| Hybrid | 0.666723 | 0.688341 | 0.689114 |

For the union outcome, event recall at a 5% frame budget is 0.238 for random,
0.242 for trajectory, 0.344 for text bigrams, 0.290 for program edge, 0.301 for
program trace, and 0.307 for hybrid. Program edge gains 0.012788 episode APFD
over random but trails raw text by 0.036095. Raw text gains 0.048884 over
random. Collision alone is less encouraging for the structured abstraction:
program edge gains only 0.004489 over random, and its tie-seed interval includes
zero.

Center containment can turn centimeter-scale map-boundary excursions into a
binary positive, so the artifact retains maximum excursion distance. In the
post-hoc, non-primary sensitivity that requires more than 10 cm outside
the drivable union, off-road positives fall from 1,550 frames/249 episodes to
862 frames/149 episodes. Program-edge episode APFD gain over random falls from
0.017627 to 0.006855; raw text remains 0.052070 above random. For the
collision-or-more-than-10-cm-off-road union, program edge is 0.005165 above
random and raw text is 0.040555 above random. Thus the qualitative result is
not semantic-program superiority:
raw text is the strongest reasoning representation, and part of the modest
structured gain is concentrated in very small boundary excursions.

The 2.5--97.5% ranges in this artifact reflect randomized greedy tie breaking
only. This suite and the outcome design were inspected adaptively, episode
definitions are not externally validated, and no population-level inference is
claimed. The trajectory baseline is also mechanically related to outcomes
derived from the same plan, although coverage diversity is not a trajectory
risk score.

### Fixed causal-obligation denominator prototype

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
excluding none/none). This produces three nested denominators:

| Arm | Frozen hierarchical items | Exact cells observed |
| --- | ---: | ---: |
| CRO applicability | 44 | 23/27 variants (85.2%) |
| CRO discharge status | 160 | 74/108 variant-status cells (68.5%) |
| CRO-conditioned decision | 2,219 | 175/1,917 variant-decision cells (9.1%) |

On the same valid, source-disjoint 14,867-frame cohort, 9,045 frames (60.8%)
trigger at least one v1 obligation. There are 9,138 triggers: 8,934 lead and
204 transverse-conflict triggers. The trace recognizes and emits the expected
decision category for 5,125 triggers; 2,221 are not recognized, 1,523 are
recognized with another decision, and 269 emit the expected category without
the corresponding recognized fact. These are parser-grounding diagnostics,
not correctness labels.

The episode-APFD bake-off is negative:

| Arm | Collision | Center off-road | Union |
| --- | ---: | ---: | ---: |
| Random | 0.656516 | 0.663268 | 0.669470 |
| Trajectory | 0.656335 | 0.664012 | 0.669999 |
| Text bigram | 0.695149 | 0.729138 | 0.718353 |
| Program path | 0.666907 | 0.683879 | 0.685897 |
| CRO applicability | 0.656944 | 0.663231 | 0.669617 |
| CRO discharge status | 0.657022 | 0.663408 | 0.669886 |
| CRO-conditioned decision | 0.657861 | 0.666308 | 0.672005 |
| CRO status + program-path hybrid | 0.667794 | 0.684159 | 0.686607 |

For the union, CRO-conditioned decisions improve episode APFD over random by
only 0.002535; the 2.5--97.5% tie-seed range of the paired delta is
[-0.000648, 0.006011]. Event recall at a 5% frame budget is 0.252 versus 0.238
for random, 0.301 for program path, and 0.344 for text bigrams. Applicability
coverage saturates all 39 cohort-observed hierarchical items after 18 frames,
and status coverage saturates all 121 after 67. The decision arm covers 271 of
its 330 cohort-observed items in the first 100 selected frames and saturates
after 159 (1.1% of the cohort). Thus the coarse denominator has too little
resolution to prioritize 14,867 frames, while the full
obligation-by-decision cross-product is mostly unsupported. Neither is a useful
middle ground yet.

A post-result diagnostic sampled 20 transverse-conflict triggers with seed
2604 and visually inspected four representative front-camera frames. The proxy
includes genuine cross traffic, but also cut-ins, encroaching or parked
vehicles, and cases where the model describes an unrelated lead or roadwork
cause. Constant-velocity closest approach establishes geometric conflict, not
right-of-way or a unique obligation to yield. The current
`crossing_agent_yield` family therefore fails semantic validation and must not
be presented as ground truth. It is retained in the versioned artifact so this
adaptive finding is auditable rather than silently tuned away.

The experiment narrows the recommended denominator. Keep the externally
triggered CRO structure, but the next frozen version should:

1. derive conflict type and applicability from map topology, traffic controls,
   route intent, actor class/kinematics, and ego path intersection;
2. encode a set of permissible response constraints rather than one expected
   decision label;
3. add criticality and temporal phase (onset, persistence, resolution) as
   bounded axes, with infeasible combinations removed before evaluation;
4. report `not exercised`, `not recognized`, `recognized with an impermissible
   response`, `reasoning/action mismatch`, and `safely discharged` separately;
   and
5. use CRO coverage as the adequacy criterion, with a preregistered secondary
   diversity rule for ties after semantic items saturate. APFD should not be
   expected to improve merely by replacing a 4,828-item lexical space with a
   small adequacy checklist.

In short, the promising denominator is not raw reasoning path and not the 27
cells in this first attempt. It is a feasible, independently triggered
obligation graph whose leaves are situation-by-permissible-response-by-phase
cells. Version 1 shows why both independent applicability and sufficient
resolution are necessary.

### Honest conclusion

There is enough signal to justify a better controlled experiment, but not to
claim semantic superiority. Raw structured coverage beats random and input or
trajectory baselines on several retrospective full-trace outcomes, yet raw
text bigrams are at least as strong, rare-item filtering removes much of the
gain, parser exactness is poor on the blinded stress audit, inferential
intervals include zero, and the effect reverses when the trace is shortened.
The prospective per-plan proxy provides stronger evidence that reasoning text
contains information useful for prioritizing unsafe emitted plans without
post-event scene aggregation, especially for center-off-road episodes. It does
not show that the current semantic parser adds value beyond wording: raw text
substantially outperforms every structured arm, collision-frame APFD is near
random, and structured gains weaken for excursions greater than 10 cm.

## Prospective confirmatory phase

The current benchmark is development data and was partly selected for semantic
coverage. Freeze the schema and greedy selection rule before evaluating a new
corpus. Split by source scene, not by frame or rollout, to prevent adjacent
samples from crossing partitions. Keep failure outcomes hidden until every arm
has produced an ordering.

Before collecting the new corpus, register a minimum useful improvement. A
reasonable starting threshold is an APFD increase of 0.05 for
`reasoning_graph` over random, specification, `image_grid`, and trajectory,
with a scene-bootstrap 95% interval above zero. Also require the direction of
the effect to remain unchanged on a fully human-reviewed coarse-label subset.
Treat hybrid improvement as a separate question: it tests whether reasoning
adds information beyond existing abstractions, not whether it replaces them.
This paragraph predates the closed-program follow-up; the concrete protocol
below supersedes it for the next experiment.

For a true online efficiency claim, generate candidate scenarios without using
outcomes, execute in batches, update coverage from newly observed traces, and
compare failures found per GPU-hour. Include the reasoning-generation and
label-extraction cost in that denominator.

### Concrete locked next experiment

Do not reuse the current OOD suite as a holdout. Use the following protocol for
the next result:

1. Repair the parser using only development text and the existing 30-trace
   audit, then run an outcome-blinded human audit on 100 additional development
   traces. Freeze the compiler, ontology, item weighting, quantile procedure,
   greedy selector, and all hashes before acquiring holdout outcomes. If
   any audited dimension is below a preregistered 80% exactness gate, stop and
   report parser failure rather than testing semantic coverage.
2. Sample 300 new unique source scenes with no source overlap with
   `benchmark.json`, the OOD suite, the unsafe-event file, or either parser
   audit. Fix inclusion and exclusions before reading outcomes.
3. Produce a fixed open-loop preview before closed-loop execution and derive
   every arm from that preview only. This avoids using a post-event reasoning
   state to prioritize its own event. Generate all 200 tie-seed orderings while
   outcome files are inaccessible, then hash and freeze them.
4. Use exhaustive closed-loop safety event as the single primary outcome and
   mean APFD delta between `program_edge` and `text_bigram` as the single
   primary contrast. Treat random, image-grid, trajectory, node, path, trace,
   hybrid, support thresholds, closed-loop failure, and recall budgets as
   secondary or exploratory. Require both a positive refit source-bootstrap
   95% interval and a preregistered minimum APFD gain of 0.05 before claiming
   that the structure adds value beyond wording.
5. Refit bins and coverage orderings inside every source-scene bootstrap
   resample, report failures found per GPU-hour including trace/compiler cost,
   and publish the locked manifest even if the result is negative.

Passing this experiment would support an operational claim about the frozen
coverage abstraction. A semantic claim would still depend on the parser audit;
APFD alone cannot prove that the covered items have the intended meaning.
