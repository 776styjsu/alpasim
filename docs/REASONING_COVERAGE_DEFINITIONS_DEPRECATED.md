# Deprecated pre-Table reasoning-coverage definitions

> [!WARNING]
> This definition is deprecated and must not be used for new Alpamayo
> reasoning-coverage results. It is retained only to interpret historical
> `reasoning_*`, `program_*`, and CRO artifacts. The canonical definition is
> [Alpamayo reasoning-coverage definitions](REASONING_COVERAGE_DEFINITIONS.md).

This document describes the coverage abstractions evaluated before the causal
reasoning obligation (CRO) prototype. Coverage is constructed without reading
collision, off-road, or manual outcome labels. Those labels are used later only
to evaluate the resulting test order with APFD and recall.

## Reasoning decomposition

The pilot contains two structured decompositions of Alpamayo reasoning: a
coarse labeled state and a deterministic reasoning program. A text-bigram arm
serves as the unstructured control.

### Coarse labeled reasoning state

Each chain of thought is labeled as

```text
(longitudinal decision,
 lateral decision,
 causal categories,
 critical tags,
 longitudinal action sequence,
 lateral action sequence)
```

For example, a trace about slowing for a lead car may contain a
`lead_obstacle_following` longitudinal decision, no lateral decision, a
`critical_objects` cause, a `lead_vehicle` tag, and a `decelerate` action.
These fields come from the benchmark reasoning label, with reviewed labels
overlaid where an audit is supplied.

The three labeled-state coverage levels are:

| Arm | Coverage items for one case |
| --- | --- |
| `reasoning_node` | The longitudinal decision, lateral decision, and each causal category. |
| `reasoning_graph` | All node items, the longitudinal/lateral decision pair, and each `cause→longitudinal decision` and `cause→lateral decision` edge. |
| `reasoning_fine` | All graph items, plus each critical tag and each atomic longitudinal or lateral action. |

Therefore `reasoning_node ⊆ reasoning_graph ⊆ reasoning_fine`. Fine coverage
distinguishes more cases, but its open-vocabulary tags are more sensitive to
labeler wording and singleton items.

### Compiled reasoning program

The program compiler reads only the chain-of-thought text. It flattens list
outputs, separates multi-step instructions, and represents each step as:

```text
facts → objectives → high-level decisions → atomic actions
  + fact location and modality
  + step entry condition
```

The full program is marked `underspecified` when any step omits either the
longitudinal or lateral decision axis; otherwise it is `resolved`. This flag is
diagnostic and does not suppress the coverage items that were successfully
compiled.

- A **fact** is a closed predicate such as `lead_slow`, `crossing_agent`,
  `signal_stop`, or `lane_blocked_or_narrow`. It also records a location such
  as `ahead_same_lane` and a modality such as `asserted`, `possible`, or
  `conditional`.
- An **objective** states the purpose, such as `maintain_following_gap`,
  `yield_right_of_way`, or `remain_in_lane_or_drivable_area`.
- A **decision** is the high-level longitudinal or lateral choice, such as
  `lead_obstacle_following`, `speed_adaptation_road_events`, `lane_change`, or
  `in_lane_nudge`.
- An **action** is an explicitly stated operation such as `decelerate`, `stop`,
  `steer_left`, or `go_straight`. Unstated actions are not inferred.
- An **entry condition** expresses sequencing, such as `immediate`,
  `when_gap_available`, or `when_path_clear`.

For example:

```text
“Decelerate to keep a safe distance from the slow lead vehicle ahead.”
```

roughly becomes:

```text
lead_present / lead_slow
  → maintain_following_gap
  → lead_obstacle_following
  → decelerate
```

The program coverage levels are:

| Arm | Coverage items for one case |
| --- | --- |
| `program_node` | Each fact, detailed `(predicate, location, modality)` fact, objective, decision, action, and entry condition. |
| `program_edge` | All node items, plus valid `fact→objective`, `objective→decision`, and `decision→action` links, and longitudinal/lateral decision and action pairs. |
| `program_path` | All edge items, plus complete `fact→objective→decision→action` paths, joint longitudinal/lateral paths, and transitions between steps within one output. |
| `program_trace` | All path items, plus decision/action transitions from the preceding frame to the current frame in the same scene. |

Thus `program_node ⊆ program_edge ⊆ program_path`. `program_trace` additionally
measures temporal changes across model calls. A complete path is emitted only
when all its components are explicitly present and linked; the compiler does
not fill missing actions merely to create a path.

#### Worked compiler examples

The following are outputs of the current compiler. The listed node, edge, and
path items are representative additions at each cumulative coverage level;
`program_edge` also contains all node items, and `program_path` also contains
all edge items.

**1. Lead-vehicle following with an explicit action**

```text
Input: Decelerate to keep a safe distance from the slow lead vehicle
       ahead in our lane.

Facts:
  lead_present | ahead_same_lane | asserted
  lead_slow    | ahead_same_lane | asserted
Objectives:
  avoid_collision_or_increase_clearance
  maintain_following_gap
Decision: lead_obstacle_following
Action:   decelerate
Entry:    immediate

Example nodes:
  program:fact=lead_slow
  program:objective=maintain_following_gap
  program:long_decision=lead_obstacle_following
  program:long_action=decelerate
Example edge:
  program:edge=lead_slow->maintain_following_gap
Example path:
  program:path_long=lead_slow>maintain_following_gap>
                    lead_obstacle_following>decelerate
```

The detailed fact is a separate node, for example
`program:fact_detail=lead_slow|ahead_same_lane|asserted`. This allows two traces
to cover the same predicate but different locations or modalities.

**2. Red-light stop**

```text
Input: Stop for the red traffic light at the intersection.

Facts:
  signal_stop          | intersection | asserted
  intersection_or_turn | intersection | asserted
Objective: obey_signal_or_stop_control
Decision:  stop_for_static_constraints
Action:    stop

Linked path:
  signal_stop > obey_signal_or_stop_control
              > stop_for_static_constraints > stop
```

`intersection_or_turn` is still covered as a fact node, but it is not used in
this path because the compiler's closed fact-objective map links
`signal_stop`, not generic intersection presence, to obeying the signal.

**3. Lateral nudge with an unstated/ambiguous atomic action**

```text
Input: Nudge left due to cones blocking the right side of our lane.

Facts:
  static_obstacle        | current_lane | asserted
  workzone_or_road_event | current_lane | asserted
Objective:        avoid_collision_or_increase_clearance
Lateral decision: in_lane_nudge
Lateral action:   not_stated

Added edge:
  avoid_collision_or_increase_clearance -> in_lane_nudge
Added complete path: none
```

Although “nudge left” sounds actionable, both `left` and `right` occur in the
sentence. The current deterministic direction parser refuses to choose one, so
it does not invent `steer_left`. The case contributes nodes and an edge but no
complete lateral path. This illustrates why path coverage is sparser than edge
coverage.

**4. Yielding without an explicit atomic action**

```text
Input: Yield to cross-traffic since vehicles are moving across our path ahead.

Facts:
  crossing_agent                    | cross_path | asserted
  yield_or_right_of_way_requirement | cross_path | asserted
Objectives:
  avoid_collision_or_increase_clearance
  yield_right_of_way
Decision: yield_agent_right_of_way
Action:   not_stated

Example edges:
  crossing_agent -> yield_right_of_way
  yield_right_of_way -> yield_agent_right_of_way
Added complete path: none
```

The high-level word “yield” determines a decision, but it does not specify
whether the atomic operation is deceleration or stopping. The compiler retains
that distinction instead of treating every yield as `stop`.

**5. Conditional lane change**

```text
Input: Change to the left lane when a safe gap opens.

Fact:             gap_available | left | conditional
Objective:        not_stated
Lateral decision: lane_change
Lateral action:   steer_left
Entry:            when_gap_available

Added edge:
  lane_change -> steer_left
Added complete path: none
```

Here the compiler captures the conditional fact and entry condition, but its
closed rules do not infer an objective from the sentence. Without a linked
fact-objective-decision chain, no full path is counted.

**6. Multi-step reasoning**

```text
Input: Change lanes to the right and decelerate for the turn,
       then accelerate once the path is clear.

Step 1:
  fact:                 intersection_or_turn | right | asserted
  objective:            adapt_speed_to_road
  decisions:            speed_adaptation_road_events + lane_change
  actions:              decelerate + steer_right
  entry:                immediate

Step 2:
  fact:                 path_clear | unspecified | conditional
  objective:            avoid_collision_or_increase_clearance
  longitudinal decision: set_speed_tracking
  longitudinal action:   accelerate
  entry:                  when_path_clear

Added within-output temporal item:
  program:temporal_long=
    speed_adaptation_road_events/decelerate>
    when_path_clear>set_speed_tracking/accelerate
```

At the next model frame, `program_trace` can add another item connecting the
last decision/action of this output to the first decision/action of the new
output. That cross-frame item is distinct from the within-output transition
shown above.

### Text baseline

`text_bigram` normalizes the reasoning text to lowercase alphanumeric words and
covers every adjacent word pair. For example, “slow lead vehicle” covers
`slow_lead` and `lead_vehicle`. It preserves much more wording detail than the
structured abstractions and tests whether apparent semantic performance can be
explained by lexical diversity alone.

## How reasoning coverage is computed

For coverage arm `a`, each case `c` produces an item set `C_a(c)`. In the
pre-CRO experiments, the denominator is the union of items observed across the
evaluated cohort:

```text
D_a = union of C_a(c) over all cases c

coverage_a(S) = distinct D_a items covered by selected cases S / |D_a|
```

The prioritizer starts with no covered items. At each step it selects the
remaining case maximizing

```text
|C_a(c) − items already covered|.
```

Equal-gain ties are randomized with a recorded seed. Once every observed item
is covered, the remaining cases are consequently ordered only by that seeded
tie order. When the evaluation unit is a scene, its item set is the union of
the item sets from all frames in that scene.

The variants ending in `support2` or `support5` remove items occurring in fewer
than two or five cases before ordering. These variants test whether results are
driven by rare or singleton text/program items; filtering also changes the
denominator.

APFD is not itself coverage. It measures how early independently labeled
failures occur in the coverage-produced order. Episode APFD counts a safety
episode as detected when the first frame belonging to it is selected.

## Other coverage baselines

| Arm | Coverage items for one case |
| --- | --- |
| `random` | No item set; a seeded random permutation. |
| `specification` | Each supplied scene attribute and every pair of attributes from different fields. |
| `visual_stats` | Quantile bins for global mean luminance, luminance variation, entropy, edge strength, and colorfulness. This is not neuron coverage. |
| `image_grid` | Quantile bins for the RGB channels in each cell of an 8×8 image grid. This is coarse input coverage, not full-resolution pixel coverage. |
| `trajectory` | Quantile bins for planned distance, lateral offset, speed, acceleration, heading change, and curvature, plus dominant longitudinal/lateral action categories and their pair. |

Continuous image and trajectory bins are fitted on the evaluated feature
cohort without outcome labels. They are therefore outcome-blind but
cohort-relative rather than externally fixed.

## Hybrid coverage

The benchmark and OOD evaluators use different hybrid unions:

| Arm | Item-set union |
| --- | --- |
| Benchmark `hybrid` | `specification ∪ trajectory ∪ reasoning_graph` |
| Benchmark `program_hybrid` | `specification ∪ trajectory ∪ program_path` |
| OOD `hybrid` | `trajectory ∪ program_trace` |

All items in a union have equal weight. A hybrid does not create cross-product
items between its components; it simply rewards any new item contributed by
any component.

## Denominator limitation

Every pre-CRO reasoning denominator is output-derived. If Alpamayo never emits
a state—or the compiler never recognizes it—that state is absent from `D_a`
instead of appearing as uncovered. These definitions consequently measure
diversity among observed outputs, not completeness against an external driving
specification.
