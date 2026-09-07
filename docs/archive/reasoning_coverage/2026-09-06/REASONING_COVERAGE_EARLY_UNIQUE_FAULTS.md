# Early and unique fault discovery with reasoning coverage

This study compares two Z1/Z2/Z3 reasoning abstractions with the available
non-reasoning coverage methods under multi-round greedy prioritization. It uses
both the 150-case human-labeled benchmark and the 14,867-frame source-disjoint
planned-safety cohort.

The main result is nuanced:

- neither reasoning abstraction improves aggregate collision discovery on the
  large cohort, but each surfaces collision frames that trajectory-grid and
  text-bigram coverage tend not to select early;
- reasoning coverage strongly front-loads off-road-positive frames, although
  continuation, text bigrams, and eventually trajectory-grid coverage discover
  more of them;
- the 150-case cohort contains no fault that is robustly exclusive to reasoning
  against the union of all available non-reasoning methods; and
- the atomic abstraction is much smaller and often matches the full abstraction,
  but the two orders are complementary. The full representation produces more
  stable full-versus-atomic exclusive discoveries, especially for off-road.

All of these are retrospective results on proxy outcomes. “Unique” means unique
to an ordering at a stated review budget, not a fault that is logically
impossible for another representation to discover.

## Coverage definitions

Both reasoning methods include Z1, Z2, and Z3.

| Arm | Coverage item |
| --- | --- |
| Full reasoning Z1/Z2/Z3 | Cumulative primary-stage Z1 components and details, Z2 roles/guard, Z3 decisions, Z1→Z2 and Z2→Z3 edges, and complete Z1→Z2→Z3 paths |
| Atomic reasoning Z1/Z2/Z3 | Only complete `Z1 component → Z2 role → Z3 axis:decision` paths from the primary stage |

A single reasoning may contribute multiple atomic paths. For example, a lane
change justified by both a slow lead vehicle and an available adjacent gap
contributes one complete path for each applicable Z1 component. It does not get
collapsed to one path per reasoning.

The atomic representation is therefore a decomposition of the conjunction,
not a version that omits Z2. Its 150-case universe contains the previously
reported 107 distinct complete Z1→Z2→Z3 paths. The full representation contains
those paths plus the corresponding nodes, details, and edges.

Observed universes and the median number of cases required to complete the
first greedy round are:

| Cohort | Arm | Items | First-round saturation |
| --- | --- | ---: | ---: |
| 150 | NuRec input-label/specification | 468 | 31 |
| 150 | Visual statistics | 20 | 5 |
| 150 | Image grid | 768 | 11 |
| 150 | Trajectory summary | 55 | 19 |
| 150 | Trajectory grid | 189 | 19 |
| 150 | Full reasoning Z1/Z2/Z3 | 364 | 56 |
| 150 | Atomic reasoning Z1/Z2/Z3 | 107 | 38 |
| 150 | Reasoning with continuation | 429 | 58 |
| 150 | Text bigram | 959 | 86 |
| 14,867 | Trajectory grid | 363 | 41 |
| 14,867 | Full reasoning Z1/Z2/Z3 | 635 | 138 |
| 14,867 | Atomic reasoning Z1/Z2/Z3 | 209 | 75 |
| 14,867 | Reasoning with continuation | 814 | 189.5 |
| 14,867 | Text bigram | 4,828 | 887 |

Multi-round additional greedy clears the covered set at saturation and starts
another coverage round. Thus every position remains coverage-directed instead
of becoming a random tail after the small item universe is exhausted. Labels
are never read while constructing an order. Randomized ties are averaged over
200 repeats on the benchmark and 50 repeats on the large cohort.

## What “unique” means here

The report uses three related comparisons:

1. **Same-budget pairwise:** a reasoning arm and one comparator each receive the
   stated budget. Candidate-only, overlap, and comparator-only counts are paired
   by tie-break repeat. This is the fairest direct comparison.
2. **Strict non-reasoning union:** a reasoning arm is compared with the union of
   all non-reasoning arms. Every comparator receives the full budget, so this is
   deliberately harder than a budget-matched portfolio and should not be read
   as one.
3. **Robustly exclusive:** the reasoning arm detects a fault in at least 75% of
   its repeats while the comparator union detects it in at most 25%.

The large-cohort non-reasoning union contains trajectory-grid and text-bigram
coverage. The benchmark union additionally contains NuRec specification,
visual-statistics, image-grid, and trajectory-summary coverage.

## Aggregate early detection and APFD

APFD averages normalized first-detection rank over all fault units; 0.5 is the
expected random value. Budget-restricted counts are more informative when a
method catches up late.

### 150 human-labeled cases

The fault unit is one case. The table gives mean faults found at the 10% budget
of 15 cases and APFD over the complete order.

| Arm | Inconsistent CoT / 26 at 10% | APFD | Unsafe / 17 at 10% | APFD |
| --- | ---: | ---: | ---: | ---: |
| Random | 2.53 | 0.503 | 1.68 | 0.496 |
| NuRec input-label/specification | 3.00 | 0.542 | 1.00 | 0.608 |
| Visual statistics | 2.48 | 0.495 | 1.65 | 0.500 |
| Image grid | 2.28 | 0.466 | 1.55 | 0.454 |
| Trajectory summary | 5.03 | 0.624 | 1.95 | 0.543 |
| Trajectory grid | **7.19** | **0.646** | **6.94** | **0.627** |
| Full reasoning Z1/Z2/Z3 | 2.21 | 0.548 | 3.08 | 0.589 |
| Atomic reasoning Z1/Z2/Z3 | 2.75 | 0.517 | 2.96 | 0.583 |
| Reasoning with continuation | 2.17 | 0.510 | 2.18 | 0.581 |
| Text bigram | 1.54 | 0.461 | 2.00 | 0.509 |

Trajectory-grid coverage is the clear benchmark leader. Atomic reasoning finds
slightly more inconsistent-CoT cases than full reasoning at 10%, while full
reasoning has better whole-order APFD and very slightly better unsafe-case
detection. These small full-versus-atomic differences are not stable enough to
declare either representation uniformly better.

The grid result should retain the qualification from the multi-round study: a
single maximum-lateral-extent score reproduces or exceeds much of its benchmark
advantage and is close to the definitions of the lateral mismatch and unsafe
labels. It is a strong prioritizer for this labeled set, not evidence of a
general semantic coverage advantage.

### Large planned-safety cohort

The fault unit is one positive 100 ms frame. NuRec semantic input labels do not
exist for this cohort, so NuRec coverage cannot be evaluated here.

| Outcome | Arm | 1% / 149 | 5% / 744 | 10% / 1,487 | APFD |
| --- | --- | ---: | ---: | ---: | ---: |
| Collision / 1,386 | Random | 13.8 | 69.4 | 140.7 | 0.500 |
|  | Trajectory grid | 6.0 | 32.2 | 72.4 | 0.520 |
|  | Full reasoning Z1/Z2/Z3 | 15.9 | 70.8 | 139.2 | 0.485 |
|  | Atomic reasoning Z1/Z2/Z3 | 16.2 | 70.1 | 143.8 | 0.474 |
|  | Reasoning with continuation | **19.5** | **76.7** | **146.5** | 0.486 |
|  | Text bigram | 16.1 | 70.1 | 139.7 | 0.476 |
| Off-road / 1,550 | Random | 15.8 | 77.1 | 156.0 | 0.500 |
|  | Trajectory grid | 55.3 | 265.6 | 427.3 | **0.723** |
|  | Full reasoning Z1/Z2/Z3 | 54.7 | 203.8 | 377.8 | 0.661 |
|  | Atomic reasoning Z1/Z2/Z3 | 47.2 | 203.2 | 379.1 | 0.651 |
|  | Reasoning with continuation | 63.1 | 265.8 | **452.1** | 0.668 |
|  | Text bigram | **68.2** | **266.6** | 452.0 | 0.679 |

No method meaningfully improves aggregate collision detection. Atomic reasoning
is four frames above random at 10%, full reasoning is two below it, and every
coverage arm has worse APFD than random except the trajectory grid's late
catch-up. The grid finds only 72.4 collision frames at 10%, roughly half the
random count, so its 0.520 APFD is not useful for an early-review claim.

Off-road frames are strongly prioritizable. Both primary-stage reasoning arms
find about 378–379 at 10%, 2.4 times random, but continuation and text coverage
find about 452. Trajectory grid has the best complete-order APFD and wins by the
25% budget because it repeatedly prioritizes long lateral plans. Atomic
reasoning retains nearly all of the primary-stage full representation's early
off-road signal with 209 rather than 635 items.

## Unique discoveries at 10%

### Strict comparison with the non-reasoning union

The columns are repeat-wise means. “Candidate only” is found by reasoning but
not by any non-reasoning comparator; “baseline only” is the reverse. The robust
column applies the 75%/25% probability rule.

| Cohort/outcome | Candidate | Candidate only | Overlap | Baseline only | Robust exclusive |
| --- | --- | ---: | ---: | ---: | ---: |
| 150 inconsistent | Full | 0.71 | 1.50 | 11.92 | 0 |
| 150 inconsistent | Atomic | 1.19 | 1.56 | 11.86 | 0 |
| 150 unsafe | Full | 0.67 | 2.40 | 7.04 | 0 |
| 150 unsafe | Atomic | 0.70 | 2.27 | 7.18 | 0 |
| Large collision | Full | 50.38 | 88.82 | 104.70 | 24 |
| Large collision | Atomic | 63.12 | 80.70 | 112.82 | 29 |
| Large off-road | Full | 68.66 | 309.14 | 440.22 | 40 |
| Large off-road | Atomic | 72.84 | 306.24 | 443.12 | 34 |

The benchmark has occasional candidate-only discoveries, but none survives the
robust definition against the six-method non-reasoning union at 5%, 10%, or
25%. This is the strongest answer available from the 150 cases: reasoning does
not expose a stable fault class missed by all other available abstractions.

The large cohort does contain stable reasoning-exclusive frames. The number is
not monotone in budget because uniqueness can disappear when comparators catch
up:

| Outcome | Reasoning candidate | 1% | 5% | 10% | 25% |
| --- | --- | ---: | ---: | ---: | ---: |
| Collision | Full | 7 | 19 | 24 | 21 |
|  | Atomic | 10 | 11 | 29 | 31 |
|  | Full ∪ atomic | 12 | 20 | 34 | 38 |
| Off-road | Full | 26 | 35 | 40 | 12 |
|  | Atomic | 20 | 25 | 34 | 17 |
|  | Full ∪ atomic | 31 | 40 | 50 | 19 |

The union row gives each reasoning arm a full budget as well, so it measures
complementarity rather than the performance of a deployable single-budget
hybrid.

### Fair same-budget pairwise comparison

At 10%, the large-cohort reasoning orders find many frames that each baseline
misses, but the baselines also find many that reasoning misses:

| Outcome | Candidate vs comparator | Candidate only | Overlap | Comparator only | Robust candidate-only |
| --- | --- | ---: | ---: | ---: | ---: |
| Collision | Full vs trajectory grid | 116.16 | 23.04 | 49.36 | 80 |
|  | Atomic vs trajectory grid | 124.22 | 19.60 | 52.80 | 81 |
|  | Full vs text bigram | 59.26 | 79.94 | 59.76 | 32 |
|  | Atomic vs text bigram | 69.54 | 74.28 | 65.42 | 33 |
| Off-road | Full vs trajectory grid | 250.34 | 127.46 | 299.84 | 191 |
|  | Atomic vs trajectory grid | 244.74 | 134.34 | 292.96 | 176 |
|  | Full vs text bigram | 119.64 | 258.16 | 193.84 | 67 |
|  | Atomic vs text bigram | 132.30 | 246.78 | 205.22 | 64 |

This is evidence of complementarity, not reasoning dominance. For off-road,
trajectory grid and text bigrams have the larger comparator-only side. For
collision, reasoning substantially improves on the trajectory grid's poor early
order, but is essentially tied with text coverage in both directions.

The benchmark shows the same pattern on a much smaller scale. At 10%, full and
atomic reasoning find 1.89 and 2.40 inconsistent cases that trajectory grid
misses, while grid finds 6.87 and 6.84 that they miss. For unsafe cases the
corresponding split is about 1.25 reasoning-only versus 5.2 grid-only. Against
NuRec, early selections are almost disjoint, but the larger union comparison
shows that other methods recover those cases.

### Full versus atomic reasoning

At the same 10% budget:

| Cohort/outcome | Full only | Overlap | Atomic only | Robust full only | Robust atomic only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 150 inconsistent | 0.56 | 1.65 | 1.10 | 0 | 0 |
| 150 unsafe | 0.52 | 2.56 | 0.41 | 0 | 0 |
| Large collision | 30.18 | 109.02 | 34.80 | 19 | 7 |
| Large off-road | 69.00 | 308.80 | 70.28 | 34 | 2 |

The mean-only sides are similar, but the probability threshold favors the full
representation. Its extra Z1/Z2/Z3 nodes, component details, and edges produce
a more stable order for some faults; the atomic order reaches a similarly sized
but less repeat-stable fringe. This does not justify discarding atomic coverage:
it has one-third as many large-cohort items, matches aggregate primary-stage
off-road detection, and contributes additional candidate-only frames.

## What the reasoning-exclusive frames look like

At 10%, the robust large-cohort reasoning-family exclusives against trajectory
grid ∪ text bigram are concentrated rather than independent:

| Outcome | Candidate | Robust frames | Source scenes | Maximum from one scene |
| --- | --- | ---: | ---: | ---: |
| Collision | Full | 24 | 11 | 4 |
|  | Atomic | 29 | 10 | 11 |
|  | Full ∪ atomic | 34 | 12 | 12 |
| Off-road | Full | 40 | 13 | 5 |
|  | Atomic | 34 | 16 | 6 |
|  | Full ∪ atomic | 50 | 18 | 7 |

The collision exclusives disproportionately describe work zones, blocked lanes,
static obstacles, and in-lane nudges while still carrying the common
lead-following obligation. In the full arm's 24 robust collision frames, 9
mention a work zone, 8 a present lead, and 7 a static obstacle; 23 have a
proximate-cause Z2 role. Atomic coverage emphasizes the same family even more:
17 of its 29 frames mention a work zone and 13 contain an in-lane-nudge Z3
decision.

Representative collision rationales include:

- “Nudge left due to vehicle merging from right.”
- “Create usable gap for right lane change because car alongside blocks merge.”
- “Keep distance to construction truck blocking lane ahead.”

The off-road exclusives emphasize intersection/turn, obstacle/nudge, gap, and
curve contexts. Of the full arm's 40 robust frames, 14 mention an intersection,
12 a lead vehicle, 9 an available gap, and 9 a slow lead; 39 contain a
proximate-cause Z2 role. Twenty-one are classified as small-lateral departures.
The atomic arm is similar: 11 intersection, 10 static-obstacle, and 9 road-curve
mentions among 34 robust frames, with 20 small-lateral departures.

Examples include a left nudge around a stopped vehicle, a left turn at an
intersection, and gap creation before a right merge. One selected off-road frame
has only about 6 cm maximum center excursion beyond the drivable union. Thus
part of the “unique fault” count is shallow boundary sensitivity rather than a
set of severe, independent safety events.

The scene concentration and frame-level outcome are the biggest caveats. For
example, 11 of atomic reasoning's 29 robust collision frames at 10% come from
one scene. A scene-level bootstrap or event-level deduplication would yield a
smaller effective sample and wider uncertainty than the tie-break bands here.

## Conclusions

1. Use multi-round greedy for this study. Single-round results on the large
   cohort mostly measure a random tail after early saturation.
2. Keep both Z1/Z2/Z3 reasoning representations. Atomic coverage is an efficient
   primary-stage abstraction; full coverage adds stable distinctions and the
   two orders find partially different faults.
3. Do not claim aggregate collision improvement. The defensible result is that
   reasoning complements trajectory-grid and text coverage by surfacing a
   different early subset.
4. For off-road review, reasoning is useful but not the best standalone method.
   A portfolio should include trajectory geometry and either continuation-aware
   reasoning or text coverage.
5. Do not claim a benchmark-exclusive reasoning fault class. All apparent
   reasoning-only discoveries in the 150 cases are recovered often enough by
   the combined non-reasoning methods to fail the robust test.
6. Re-evaluate uniqueness at the scene or failure-episode level before making a
   safety-effectiveness claim. The present large-cohort numbers count dependent
   100 ms frames and include shallow off-road crossings.

## Artifacts and reproduction

The machine-readable report, complete per-fault probabilities, and curves are:

- `runs/reasoning_coverage/experiment_analysis_multiround_atomic.json`
- `runs/reasoning_coverage/experiment_analysis_multiround_atomic.benchmark_early_faults.csv`
- `runs/reasoning_coverage/experiment_analysis_multiround_atomic.planned_early_faults.csv`
- `runs/reasoning_coverage/experiment_analysis_multiround_atomic.benchmark_curves.csv`
- `runs/reasoning_coverage/experiment_analysis_multiround_atomic.planned_curves.csv`
- `runs/reasoning_coverage/figures_multiround_atomic/`

Reproduce them from the repository root with:

```bash
PYTHONPATH=src/utils:src/tools .venv/bin/python \
  -m reasoning_coverage.experiment_analysis \
  --multi-round \
  --benchmark-repeats 200 \
  --planned-repeats 50 \
  --workers 16 \
  --output runs/reasoning_coverage/experiment_analysis_multiround_atomic.json \
  --figure-dir runs/reasoning_coverage/figures_multiround_atomic
```

The large-cohort run excludes visual-statistics, image-grid, and
trajectory-summary controls because exact multi-round ordering is
computationally pathological for their many distinct signatures and very small,
frequently resetting item universes. All are included for the 150-case cohort.
NuRec label coverage is also benchmark-only because equivalent labels are not
stored for the large cohort.
