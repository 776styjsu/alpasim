# Multi-round greedy prioritization

This report re-runs the cross-cohort experiment with multi-round greedy
prioritization and compares it against the single-round protocol used in
[the main experiment analysis](REASONING_COVERAGE_EXPERIMENT_ANALYSIS.md).
The follow-up [early and unique fault study](REASONING_COVERAGE_EARLY_UNIQUE_FAULTS.md)
adds atomic Z1→Z2→Z3 paths and per-fault exclusivity comparisons under this
multi-round protocol.

The change is confined to test ordering. Coverage construction, cohorts, budgets,
seeds, and repeat counts are unchanged, and ordering remains outcome-blind.

The large-cohort fault unit is one positive frame. An earlier version of this
report merged runs of consecutive positive decisions into episodes; that unit
has been dropped, and
[what the frame unit costs reasoning coverage](#what-the-frame-unit-costs-reasoning-coverage)
records what changed with it.

## Why the ordering rule was confounding the comparison

Single-round greedy stops being coverage-directed once no remaining case adds an
uncovered item; the rest of the order is the seeded random tail. On the
source-disjoint planned-safety cohort, that happens almost immediately:

| Arm | Observed items | Saturates after | Share of cohort |
| --- | ---: | ---: | ---: |
| Trajectory grid | 363 | 41 cases | 0.28% |
| Reasoning Z1/Z2/Z3 | 635 | 138 cases | 0.93% |
| Reasoning with continuation | 814 | 190 cases | 1.27% |
| Text bigram | 4,828 | 887 cases | 5.97% |

At the headline 10% budget of 1,487 frames, the trajectory-grid order was
therefore 97% random filler and the reasoning orders about 91%. The arms were
being compared mostly on their random tails, and the one arm still discriminating
at that budget — text bigram, by virtue of a 4,828-item universe — was
structurally advantaged. On the 150-case benchmark the same universes saturate
between 12.7% and 57.3% of the cohort, so that cohort was much less affected.

Multi-round greedy clears the covered set whenever coverage saturates and keeps
selecting, so round `r` covers each observed item for the `r`-th time and every
position in the order stays coverage-directed. This is the standard "additional
greedy with reset" rule from test prioritization. It removes the universe-size
advantage and, more importantly, exposes each abstraction's own bias instead of
diluting it with random selection.

## Results

Mean positive frames detected, single-round (S) versus multi-round (M), averaged
over the same tie-break repeats. The fault unit is one positive frame, so the
random APFD baseline is 0.5.

### Collision-positive frames (1,386, source-disjoint planned safety)

| Arm | 1% S | 1% M | 5% S | 5% M | 10% S | 10% M | 25% S | 25% M | APFD S | APFD M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 13.8 | 13.8 | 69.4 | 69.4 | 140.7 | 140.7 | 346.0 | 346.0 | 0.500 | 0.500 |
| Trajectory grid | 12.3 | 6.0 | 67.9 | 32.2 | 139.3 | 72.4 | 345.2 | 267.4 | 0.500 | 0.520 |
| Reasoning Z1/Z2/Z3 | 15.8 | 15.9 | 71.7 | 70.8 | 142.4 | 139.2 | 347.3 | 335.1 | 0.501 | 0.485 |
| Reasoning + continuation | **19.5** | **19.5** | **76.6** | **76.7** | **146.8** | **146.5** | **351.4** | 334.7 | **0.503** | 0.486 |
| Text bigram | 16.1 | 16.1 | 70.1 | 70.1 | 140.3 | 139.7 | 349.6 | 334.1 | 0.501 | 0.476 |

Multi-round ordering does not rescue collision detection; it makes it worse.
Reasoning and text bigrams drop slightly below random, and the trajectory grid
detects roughly half as many collision frames as random through the whole useful
range.

The grid's multi-round APFD of 0.520 is misleading. It trails random at every
budget through 25% and overtakes it only between 25% and 40% (603 versus 549
frames at 40%, 770 versus 689 at 50%). APFD integrates the entire order, so a
late catch-up outweighs a deficit across every budget a reviewer would use.
Budget-restricted recall is the operative number here.

### Center-off-road-positive frames (1,550)

| Arm | 1% S | 1% M | 5% S | 5% M | 10% S | 10% M | 25% S | 25% M | APFD S | APFD M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 15.8 | 15.8 | 77.1 | 77.1 | 156.0 | 156.0 | 386.3 | 386.3 | 0.500 | 0.500 |
| Trajectory grid | 29.2 | 55.3 | 89.9 | 265.6 | 167.3 | 427.3 | 394.5 | **879.0** | 0.504 | **0.723** |
| Reasoning Z1/Z2/Z3 | 51.8 | 54.7 | 111.9 | 203.8 | 188.0 | 377.8 | 411.4 | 778.5 | 0.511 | 0.661 |
| Reasoning + continuation | 63.1 | 63.1 | 138.8 | **265.8** | 213.0 | **452.1** | 432.2 | 778.7 | 0.520 | 0.668 |
| Text bigram | **68.2** | **68.2** | **266.6** | 266.6 | **361.7** | 452.0 | **557.4** | 797.8 | **0.569** | 0.679 |

Every arm improves substantially, and the ranking depends on the budget. Text
bigrams lead at 1%; three techniques tie at 5%; continuation-aware reasoning and
text bigrams tie at 10%; and the trajectory grid wins by 25% and on APFD. The
grid's late advantage is duration weighting — it front-loads long lateral
excursions, and each contributes many positive frames.

### A confound that turns out to be a saturation artifact

The single-round analysis attributed most of the text-bigram off-road advantage
to lexical rarity: removing singleton bigrams dropped frame APFD from 0.569 to
0.528, eliminating 59% of the lift over random. Under multi-round ordering the
support-≥2 arm reaches 0.676 against the raw arm's 0.679 — removing singletons
now costs under 2% of the lift. The rarity confound existed because the
single-round order was mostly a random tail, not because text coverage depends
on singletons.

What survives is that scalar diagnostics still match full bigram coverage:
inverse bigram support alone reaches 0.683 APFD and CoT length alone 0.611,
against multi-round text-bigram coverage's 0.679.

### 150-case benchmark

Multi-round ordering is a no-op here through the reported budget range, by
construction. The earliest any arm saturates is 19 cases (trajectory grid; the
others at 31, 56, 58, and 86), so at the 5% budget of 8 cases and the 10% budget
of 15 cases no arm has saturated and the two rules produce the same orders.

| Arm | 5% (8 cases) S→M | 10% (15) S→M | 25% (38) S→M |
| --- | ---: | ---: | ---: |
| **Inconsistent reliable CoT (26)** | | | |
| Random | 1.42 → 1.42 | 2.53 → 2.53 | 6.62 → 6.62 |
| NuRec scene specification | 2.00 → 2.00 | 3.00 → 3.00 | 6.78 → 6.00 |
| Trajectory grid | **5.54 → 5.54** | **7.19 → 7.19** | **12.07 → 12.40** |
| Reasoning Z1/Z2/Z3 | 1.45 → 1.45 | 2.21 → 2.21 | 7.90 → 7.90 |
| Reasoning + continuation | 1.00 → 1.00 | 2.17 → 2.17 | 6.35 → 6.35 |
| Text bigram | 0.54 → 0.54 | 1.54 → 1.54 | 5.35 → 5.35 |
| **Unsafe trajectory (17)** | | | |
| Random | 0.93 → 0.93 | 1.68 → 1.68 | 4.20 → 4.20 |
| NuRec scene specification | 1.00 → 1.00 | 1.00 → 1.00 | 4.74 → 5.08 |
| Trajectory grid | **4.46 → 4.46** | **6.94 → 6.94** | **8.94 → 8.12** |
| Reasoning Z1/Z2/Z3 | 2.45 → 2.45 | 3.08 → 3.08 | 5.39 → 5.39 |
| Reasoning + continuation | 1.00 → 1.00 | 2.18 → 2.18 | 5.40 → 5.40 |
| Text bigram | 1.00 → 1.00 | 2.00 → 2.00 | 5.45 → 5.45 |

At the 25% budget of 38 cases only the two arms that have saturated by then —
trajectory grid and scene specification — change at all. The saturation confound
was specific to the large cohort, and this cohort's conclusions stand as
published: the trajectory grid dominates, beating random in 100% of tie-break
repeats on both outcomes (+0.122 APFD on inconsistent CoT, +0.164 on unsafe).

Two qualifications matter for how much that result should carry.

**The grid's advantage here reduces to a single geometric scalar.** Ranking
cases by maximum lateral extent alone — one outcome-blind number per case, no
coverage structure — reaches 0.655 APFD and 7.76 inconsistencies at 15 cases,
which is *better* than the full 189-item grid coverage (0.625, 7.19), and
0.619 versus 0.660 on unsafe trajectories. Cell count and maximum forward reach
alone are at or below random (0.474 and 0.448 APFD). So the operative signal is
specifically lateral deviation, and the grid's coverage construction adds
nothing on top of it. The 60 cases whose plan leaves the central lateral cell
carry a 26.7% inconsistency rate and a 16.7% unsafe rate, against 10.1% and 7.9%
for the 89 near-straight cases.

Given the label composition documented in the main analysis — 17 of the 26
inconsistencies are lateral maneuver or direction mismatches, and the unsafe
label is a human judgment about the emitted plan — a lateral-extremity ranker is
close to restating the labels' own definition. The definitions document already
flags this circularity for the unsafe outcome; it applies to the
CoT-inconsistency outcome as well.

**Reasoning coverage is not reducible the same way, but this cohort cannot
resolve it.** Its prioritization signal does come from structure rather than
size: reasoning-path coverage reaches 0.542 APFD on inconsistent CoT where
ranking by reasoning item count reaches only 0.454. But against random its
paired advantage is +0.039 APFD on inconsistent CoT (95% tie-break band
[-0.022, +0.102], ahead in 86% of repeats) and +0.072 on unsafe trajectories
(band [-0.023, +0.155], 94% of repeats). Both bands cross zero. With 26 and 17
faults, this cohort has no power to separate reasoning coverage from random in
either direction, and the bands quantify tie-breaking only — a scene-level
resample would be wider still.

## Why the ordering rule changes the ranking

Multi-round ordering amplifies each abstraction's bias rather than diluting it,
which makes the biases measurable.

**The trajectory grid is a plan-geometry ranker.** Its greedy rule prefers cases
covering many cells, and in every round it must re-cover the rare far-lateral
cells. Its 10%-budget selection has a mean maximum forward reach of 17.40 cells
against a cohort mean of 8.21, and a mean maximum lateral extent of 1.72 cells
against a cohort mean of 0.27 — a 6.4-fold lateral skew. The selection is
therefore 28.63% off-road-positive against 10.43% prevalence, and only 4.83%
collision-positive against 9.32% prevalence. Because the off-road label is itself
a geometric property of the emitted plan, this sharpens the circularity already
noted in the definitions document: the grid and that outcome are both functions
of plan geometry. Collision is not a plan-geometry property, and the grid
actively deprioritizes it.

Because the fault unit is the frame, this bias is rewarded on off-road: the long
lateral plans the grid promotes each contribute many positive frames, which is
why it reaches the highest off-road APFD (0.723) and the best 25%-budget recall
despite being no better than the alternatives at 5% and 10%.

**Reasoning coverage is not a plan-geometry proxy.** Its multi-round 10%
selection has a mean grid-cell count of 9.45 against a cohort mean of 9.43 — it
does not skew toward long plans at all — yet it raises off-road prevalence in the
selection to 25.35% against 10.43% in the cohort. Whatever signal it carries is
not trajectory shape. It is also not a size proxy: on the 150-case cohort,
reasoning-path coverage reaches 0.542 APFD on inconsistent CoT where ranking by
reasoning item count reaches only 0.454.

## What the frame unit costs reasoning coverage

The fault unit was changed from the 100 ms-contiguous episode to the individual
positive frame after these runs. That change removes the metric under which
reasoning coverage performed best, and the reason is mechanical.

Under episode merging, multi-round reasoning ordering reached 45.5 distinct
collision episodes at a 10% budget against random's 42.2, while selecting
slightly *fewer* positive frames than random (137.7 versus 140.0) — it spread its
selections across separate events instead of returning many frames of the same
one. On off-road it converted positive frames into distinct episodes most
efficiently of any arm (3.27 frames per episode, against 4.37 for the trajectory
grid). The frame unit gives no credit for that and full credit for duration, so
reasoning's advantage disappears and the grid's concentration on a few long
excursions becomes its main asset.

Neither unit is unconditionally right. The frame unit matches the object a
reviewer inspects and has a clean 0.5 random baseline; the episode unit matches
the question "how many separate defects did this budget find". The results above
answer the first question only, and no claim about distinct-fault discovery
should be read off them.

## Status of the earlier conclusions

- "The trajectory grid is indistinguishable from random for collision" was a
  random-tail artifact. Under multi-round ordering the abstraction actively
  anti-prioritizes collision frames at every realistic budget.
- No arm prioritizes collision-positive frames. This is the clearest negative
  result in the study and it survives both ordering rules.
- "Text bigrams are the strongest large-cohort method after the earliest
  budgets" holds for off-road only up to about a 10% budget, after which the
  trajectory grid leads.
- The lexical-rarity confound attributed to singleton bigrams was itself a
  single-round artifact and largely disappears under multi-round ordering.
- The 150-case conclusions are unchanged, and multi-round ordering cannot change
  them: no arm saturates before the 15-case budget. "Trajectory grid is the
  strongest method on the 150-case cohort" holds, but that cohort cannot
  distinguish reasoning coverage from random, and the grid's advantage on it is
  reproduced by a one-dimensional lateral-extent score that is near-circular
  with the label definitions.

These are still retrospective prioritization results on open-loop proxy
outcomes, and the intervals still describe tie-breaking rather than corpus
uncertainty. Multi-round ordering removes one confound; it does not address the
lexical-rarity confound, the plan-derived nature of the off-road label, or the
absence of a scene-level bootstrap.

## Reproduction

```bash
uv run --project src/tools python -m reasoning_coverage.experiment_analysis \
  --multi-round \
  --benchmark-repeats 200 \
  --planned-repeats 50 \
  --workers 16 \
  --output runs/reasoning_coverage/experiment_analysis_multiround.json \
  --figure-dir runs/reasoning_coverage/figures_multiround
```

Omitting `--multi-round` reproduces
`runs/reasoning_coverage/experiment_analysis.json` exactly; that equivalence was
verified numerically across every curve, coverage, and per-fault value in the
report.
