# Reasoning-coverage experiment analysis

This analysis compares random ordering, NuRec scene specification coverage,
trajectory-grid coverage, Alpamayo Z1/Z2/Z3 reasoning coverage, reasoning with
explicit continuations, and text-bigram coverage.

Every fault unit here is a single test case: one `clip_id` in the 150-case
benchmark, one emitted plan in the large cohort. The large cohort previously
merged runs of consecutive positive decisions into episodes; that unit has been
dropped, and the consequences are discussed under
[What the frame unit changes](#what-the-frame-unit-changes).

The main results are:

- trajectory-grid coverage is the strongest method on the 150-case human-label
  cohort, but its advantage there is reproduced by a single geometric scalar;
- no technique meaningfully prioritizes collision-positive frames at any
  realistic review budget; and
- off-road-positive frames are strongly prioritizable, and which technique wins
  depends on the ordering rule and the budget.

There is no evidence that reasoning coverage finds a class of faults that every
comparator systematically misses.

## Evaluation protocol

Coverage is constructed without reading fault labels. For every technique,
greedy ordering selects the case with the largest uncovered-item gain and uses
random tie-breaking. Results are averaged over 200 tie-break repeats for the
150 cases and 50 repeats for the large cohort.

The 150-case analysis contains 26 inconsistent reliable CoTs and 17 unsafe
trajectories. The large analysis uses 14,867 valid, source-disjoint frames
containing 1,386 collision-positive frames and 1,550 center-off-road-positive
frames.

Shaded curve intervals describe tie-breaking variation only; they are not
confidence intervals over independently sampled driving corpora. Adjacent frames
within a scene are strongly dependent, so a technique that returns many frames
from one long failure scores the same as one that returns single frames from as
many distinct failures.

Results are reported under both greedy rules. Single-round ordering stops being
coverage-directed once an arm's observed item universe is saturated, which on
the large cohort happens after 0.28%–5.97% of the budget; multi-round ordering
clears the covered set at saturation and keeps selecting. See
[multi-round greedy prioritization](REASONING_COVERAGE_MULTIROUND.md).

## Faults detected over budget

![150-case fault-detection curves](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/figures/benchmark150_fault_detection_curves.png)

![Large-cohort frame-detection curves](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/figures/planned_safety_fault_detection_curves.png)

Mean fault units detected at a 10% budget — 15 cases for the benchmark and
1,487 frames for planned safety — under single-round (S) and multi-round (M)
ordering:

| Technique | Inconsistent CoT / 26 | Unsafe / 17 | Collision frames / 1,386 | Off-road frames / 1,550 |
| --- | ---: | ---: | ---: | ---: |
| Random | 2.53 | 1.68 | 140.7 | 156.0 |
| NuRec scene specification | 3.00 | 1.00 | unavailable | unavailable |
| Trajectory grid | **7.19** | **6.94** | 139.3 → 72.4 | 167.3 → 427.3 |
| Reasoning Z1/Z2/Z3 | 2.21 | 3.08 | 142.4 → 139.2 | 188.0 → 377.8 |
| Reasoning with continuation | 2.17 | 2.18 | **146.8** → 146.5 | 213.0 → **452.1** |
| Text bigram | 1.54 | 2.00 | 140.3 → 139.7 | **361.7** → 452.0 |

Positive-frame APFD over the full order:

| Technique | Collision S | Collision M | Off-road S | Off-road M |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.500 | 0.500 | 0.500 | 0.500 |
| Trajectory grid | 0.500 | 0.520 | 0.504 | **0.723** |
| Reasoning Z1/Z2/Z3 | 0.501 | 0.485 | 0.511 | 0.661 |
| Reasoning with continuation | **0.503** | 0.486 | 0.520 | 0.668 |
| Text bigram | 0.501 | 0.476 | **0.569** | 0.679 |

### Collision frames are not prioritizable

Under single-round ordering every arm is within 0.003 APFD of random, and the
best 10%-budget result is 146.8 frames against random's 140.7 — a 4% relative
gain that sits inside the tie-break interval. Multi-round ordering does not
rescue this: reasoning and text bigrams fall slightly *below* random (0.485,
0.486, 0.476), and the trajectory grid detects roughly half as many
collision frames as random through the entire useful range (32.2 versus 69.4 at
5%, 72.4 versus 140.7 at 10%, 267.4 versus 346.0 at 25%).

The trajectory grid's multi-round APFD of 0.520 is an artifact of whole-order
integration: it trails random at every budget through 25% and only overtakes it
between 25% and 40% (603 versus 549 frames at 40%). At any budget a reviewer
would actually use, it is the worst arm. This is why budget-restricted recall is
reported alongside APFD and takes precedence when the two disagree.

The characterization explains the negative result. Collision-positive frames
concentrate in the most ordinary situations: a lead-following obligation is
present in 1,096 of 1,386 frames (79%), a light vehicle in 968 (70%), and the
single most common reasoning signature covers 31.2% of them. Greedy coverage
prioritizes *rare* items, so it cannot preferentially surface faults that live
in the corpus mode.

### Off-road frames are strongly prioritizable

Off-road detection improves by a factor of 2.7 over random at a 10% budget, and
the ordering rule matters more than the abstraction:

| Technique | 1% | 5% | 10% | 25% |
| --- | ---: | ---: | ---: | ---: |
| Random | 15.8 | 77.1 | 156.0 | 386.3 |
| Trajectory grid (multi) | 55.3 | 265.6 | 427.3 | **879.0** |
| Reasoning Z1/Z2/Z3 (multi) | 54.7 | 203.8 | 377.8 | 778.5 |
| Reasoning + continuation (multi) | 63.1 | 265.8 | **452.1** | 778.7 |
| Text bigram (multi) | **68.2** | **266.6** | 452.0 | 797.8 |

Text bigrams lead at the smallest budget, three techniques are tied at 5%,
continuation-aware reasoning and text bigrams are tied at 10%, and the
trajectory grid wins by 25%. The grid's late advantage comes from duration
weighting: it front-loads long lateral excursions, and each such plan
contributes many positive frames.

### Why text bigrams prioritize off-road outcomes

![Text-bigram off-road diagnostic](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/figures/text_bigram_offroad_diagnostic.png)

Off-road-positive frames are lexically distinctive. They average 18.34 words and
17.02 unique bigrams against 12.98 and 11.95 for other valid frames, and contain
1.065 bigrams with corpus support at most two against 0.160 elsewhere — a
6.7-fold difference. Of the 4,828-item bigram universe, 2,411 items occur once.

Under single-round ordering this rarity was doing most of the work: removing
singleton bigrams dropped off-road frame APFD from 0.569 to 0.528, eliminating
59% of the lift over random. **That confound is itself a saturation artifact.**
Under multi-round ordering the support-≥2 arm reaches 0.676 against the raw
arm's 0.679, so removing singletons costs under 2% of the lift. Rare wording
was standing in for the coverage signal only while the order was mostly a random
tail.

What does survive is that a one-dimensional outcome-blind score matches full
bigram coverage: ranking by inverse bigram support alone reaches 0.683 APFD, and
CoT length alone reaches 0.611. Text-bigram coverage adds nothing over a scalar
rarity score.

The associated language is not arbitrary — `then match` occurs in 216 off-road
frames of 259 occurrences, `after merging` in 213 of 264, and `change to` in 236
of 376, against 10.4% cohort prevalence. These phrases describe lateral, merge,
and gap-management plans whose trajectories are more likely to cross the
drivable-area boundary. Adjacent frames and phrases from the same scenes are
correlated, so this does not show that rare wording causes unsafe plans or that
the advantage transfers to another model or corpus.

Collision-positive frames have no comparable lexical separation: their mean
singleton-bigram count is 0.190 versus 0.159 elsewhere, and raw text-bigram
frame APFD is 0.501.

## Where the faults occur and what they involve

![Fault characterization](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/figures/fault_characterization.png)

### 150 human-labeled cases

The 26 CoT/action inconsistencies are primarily lateral: 17 are lateral maneuver
or direction mismatches, five are longitudinal-control mismatches, two are
maneuver-magnitude mismatches, and two concern a premature or missing
continuation.

The 17 unsafe trajectories comprise five agent/obstacle collision or clearance
problems, four road-edge/off-road excursions, four wrong-way or opposing-lane
maneuvers, one wrong-lane/merge-facility error, one traffic-control violation,
and two unspecified justifications. These categories are deterministic keyword
codes over the existing human justifications and should be reviewed before
publication.

NuRec context indicates that intersections contain 15 of 26 inconsistencies and
9 of 17 unsafe trajectories. Relative to their cohort support, left turns have
2.62 times the cohort inconsistency rate, right turns 2.10 times the
inconsistency rate and 3.21 times the unsafe rate, and intersection layouts 1.55
times the inconsistency rate and 1.42 times the unsafe rate. These are
exploratory ratios with small numerators, not population-risk estimates.

### Large planned-safety cohort

Collision frames are actor-centered. A lead-following obligation is present in
1,096 of 1,386 frames and a near-distance obligation in 685; a light vehicle
appears in 968, a heavy vehicle in 164, and a vulnerable road user in 11. These
actors are contextual exposures, not confirmed collision counterparts, because
the stored collision label is Boolean and does not retain actor identity.

Off-road frames are less actor-centered: 891 of 1,550 carry no high-confidence
actor obligation. Their reasoning mentions an intersection or turn in 339 frames,
a work zone or road event in 289, a merge/split/ramp in 116, and a curve in 84;
a static obstacle appears in 379. These counts are derived from the CoT compiler
and therefore characterize the reasoning abstraction, not independently labeled
road geometry.

The geometric location of first center departure is:

- median onset 1.3 s into the 3 s horizon, with a 0.3 s–2.7 s interdecile range;
- median forward distance 6.80 m from the decision origin (p90 20.29 m);
- 347 frames on the ego-left side, 303 on ego-right, and 900 with at most 0.25 m
  ego-relative lateral offset; and
- median maximum center excursion only 0.135 m outside the drivable union,
  versus 2.696 m at the 90th percentile.

Many binary off-road faults are therefore shallow boundary crossings. The
existing greater-than-10-cm sensitivity outcome remains important when making
safety claims.

### Faults on the trajectory grid

![150-case faults on the ego-centric trajectory grid](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/figures/benchmark150_fault_grid_heatmaps.png)

![Trajectory-grid versus reasoning coverage over 150-case fault cells](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/figures/benchmark150_coverage_grid_heatmaps.png)

The 26 reliable-CoT/action inconsistencies cross 140 distinct cells, and the 17
unsafe-trajectory cases cross 130. At the 10% budget of 15 cases,
trajectory-grid prioritization detects 7.19 inconsistency cases on average
versus 2.21 for reasoning, and 6.94 unsafe cases versus 3.08.

![Positive frames on the ego-centric trajectory grid](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/figures/planned_safety_fault_grid_heatmaps.png)

![Trajectory-grid versus reasoning coverage over fault cells](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/figures/planned_safety_coverage_grid_heatmaps.png)

Each square is one origin-centered 2 m by 2 m cell in the ego rig frame. The
plans of collision-positive frames cross only 81 distinct cells, while off-road
plans cross 326 — collision faults sit in a narrow, well-travelled forward
corridor and off-road faults are spatially diffuse. That contrast is the spatial
statement of why the grid is useless for collision and effective for off-road.
The third panel places each off-road frame at its exact first center departure,
covering 101 cells; the collision artifact does not store a collision waypoint,
so no equivalent panel exists.

## How faults are distributed over reasoning paths

| Fault outcome | Faults | Unique reasoning signatures | Largest-signature share | Effective signatures |
| --- | ---: | ---: | ---: | ---: |
| Inconsistent reliable CoT | 26 | 19 | 15.4% | 16.6 |
| Unsafe trajectory | 17 | 15 | 11.8% | 14.4 |
| Collision frame | 1,386 | 112 | 31.2% | 17.2 |
| Off-road frame | 1,550 | 240 | 13.9% | 54.0 |

“Effective signatures” is `exp(Shannon entropy)`. Collision frames are markedly
concentrated — one signature covers nearly a third of them, and 1,386 frames
reduce to 17.2 effective signatures — while off-road frames spread over 54.0.
Collision frames occur in 33 source scenes, with the largest contributing 7.6%
and an effective scene count of 23.5; off-road frames occur in 56 scenes, the
largest contributing 5.1%, with an effective count of 43.0.

## What reasoning finds more effectively

![Reasoning value by fault stratum](../../../../experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage/figures/reasoning_stratum_value_at_10pct.png)

At a 10% budget under single-round ordering, primary reasoning has a positive
mean detection-probability delta over the trajectory grid on both outcomes
(+0.0022 collision, +0.0133 off-road) and over random (+0.0012, +0.0206), and it
loses to text bigrams on off-road (-0.1121). Under multi-round ordering it gains
sharply against random on off-road (+0.1431) but now loses to the trajectory
grid (-0.0319) and to continuation-aware reasoning (-0.0479).

Under a deliberately strict definition — reasoning detection probability at
least 0.75 and every non-random comparator at most 0.25 — there are **zero
reasoning-only faults** at both 10% and 25% budgets, in either cohort, under
either ordering rule. The defensible conclusion is pairwise and mechanistic:
structured reasoning helps prioritize some relational situations that trajectory
shape misses, but the experiment does not establish exclusive fault-discovery
power.

## What the frame unit changes

Counting each positive frame separately weights a failure by its duration. The
earlier episode-merged analysis asked a different question — how many *distinct*
failures a budget reaches — and the two units rank the techniques differently.
Under episode merging, reasoning coverage reached more distinct collision
failures than random at a 10% budget while selecting slightly fewer positive
frames, because it spread its selections across separate events. The frame unit
does not reward that behavior, so the advantage does not appear here, and the
trajectory grid's habit of returning many frames from a few long lateral
excursions becomes its largest asset on off-road.

Neither unit is unconditionally correct. The frame unit matches the object a
reviewer inspects and gives a clean 0.5 random baseline; the episode unit
matches the question "how many separate defects did we find". Any claim about
distinct-fault discovery needs an event-level unit and should not be read off
these curves.

## Reproduction and artifacts

```bash
uv run --project src/tools python -m reasoning_coverage.experiment_analysis \
  --benchmark-repeats 200 \
  --planned-repeats 50 \
  --workers 16
```

Add `--multi-round` for the multi-round ordering rule.

Outputs:

- `runs/reasoning_coverage/experiment_analysis.json`: full curves,
  characterization, per-stratum comparisons, and per-fault advantage records;
- `runs/reasoning_coverage/experiment_analysis.benchmark_curves.csv`;
- `runs/reasoning_coverage/experiment_analysis.planned_curves.csv`;
- `runs/reasoning_coverage/experiment_analysis.spatial_grid.csv`, containing
  large-cohort per-cell fault support and 10% detection probabilities;
- `runs/reasoning_coverage/experiment_analysis.benchmark_spatial_grid.csv`; and
- `runs/reasoning_coverage/figures/`: publication-ready PNG figures.
