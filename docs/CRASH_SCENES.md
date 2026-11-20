# Forced-crash scenes

Tools for producing labelled crashes in AlpaSim, to test safety monitors such as
the two-second monitor in `/project/AV/monitors/`. The code is in
[src/tools/crash_scenes/](../src/tools/crash_scenes/); the launchers are in
[tools/](../tools/).

## Why this works

A recorded scene replays its actors along their recorded paths. With traffic
simulation off (the default) they do not react to the ego, and AlpaSim has no
contact physics. So if the ego leaves its recorded path and drives into another
actor, the two boxes overlap and nothing stops them. A crash is a matter of
geometry: pick an ego path that meets an actor's recorded path.

Each crash comes with two matched drives of the same scene, so that a monitor
can be scored on false alerts as well as on missed crashes:

- **near miss** — the same maneuver, scaled down until the ego passes the crash
  target about 0.5 m clear;
- **control** — the recorded drive, with no maneuver.

## Pipeline

```
mine ──▶ plans ──▶ run in AlpaSim ──▶ confirm
(geometry)  (choose)   (controller + vehicle model)   (label, compare with plan)
```

All commands run from the repository root. `PY` below stands for
`PYTHONPATH=src/utils:src/tools .venv/bin/python`.

1. **Mine.** Search every cached scene for ego maneuvers that end in a crash.
   CPU only; no simulator runs.

   ```bash
   sbatch --export=ALL,OUTPUT_DIR=experiments/crash_scenes/mine_<date> tools/mine_crash_scenes.slurm.sh
   ```

   Writes `manifest.jsonl` (one planned crash per row), `summary.json` (counts by
   actor group, struck ego zone, crash type and maneuver) and
   `search_config.json`.

2. **Choose crashes and write plans.**

   ```bash
   $PY -m crash_scenes plans --mine-dir experiments/crash_scenes/mine_<date> \
       --output-dir experiments/crash_scenes/plans_<name> --per-cell 3 --max-per-scene 4
   ```

   A cell is (actor group, struck ego face, crash type). Cells are filled rarest
   first, so cyclists and trucks are not crowded out by cars.

3. **Run and confirm.**

   ```bash
   sbatch --export=ALL,PLAN_SET=experiments/crash_scenes/plans_<name> tools/run_crash_scenes.slurm.sh
   ```

   Runs every plan file in AlpaSim, then writes `confirmation.jsonl` next to the
   plans. Finished runs are skipped on a re-run.

Two more commands work on any rollout, forced or not:

```bash
# What did the ego hit, and where? One JSON row per rollout.
$PY -m crash_scenes label <run dir or rollout.asl> --output labels.jsonl

# Run the two-second monitor over saved rollouts instead of live.
$PY -m crash_scenes replay-monitor <run dir or rollout.asl> --output-dir <dir>
```

## Crash labels

`label` reads `rollout.asl` and reports every actor the ego touches after the
policy takes over, ordered by time of first contact. Only the first is
physically meaningful: afterwards the ego keeps driving through whatever it hit.

Because the boxes pass through each other, the label is taken at the *first
instant* of contact, found by scanning between the logged 0.1 s steps (10 ms
scan, refined to 1 ms). At high closing speed the first logged overlap can be
metres deep and point at the wrong side of the car.

| Field | Meaning |
| --- | --- |
| `actor_id`, `actor_label`, `actor_is_static` | The actor hit, its scene class label, and whether it is a parked/static track. |
| `actor_group` | `car` (automobile, other_vehicle), `truck` (heavy_truck, trailer), `bus` (bus, train_or_tram_car), `cyclist` (rider), `pedestrian` (person, stroller), `animal`, `object` (protruding_object). The scene label `rider` does not separate bicycles from motorcycles. |
| `time_us` | First contact. |
| `ego_zone8`, `ego_zone4`, `ego_contact_uv` | Where the ego was hit (below). |
| `actor_zone8`, `actor_zone4`, `actor_contact_uv` | Where the other actor was hit. |
| `crash_type` | Derived from the two struck faces and the heading difference (below). |
| `striking` | `ego`, `actor`, `both` or `neither`: whose own motion carried it into the contact point (faster than 0.5 m/s). |
| `ego_speed_mps`, `actor_speed_mps`, `closing_speed_mps`, `heading_difference_deg` | Kinematics at contact. |
| `overlap_duration_s` | How long the boxes stay overlapped. |
| `max_penetration_m` | Deepest the boxes get into each other while the ego drives through. A corner graze stays shallow; use this to filter. |
| `overlap_at_start` | The boxes already overlapped when labelling began, so there is no moment of first contact. |

**Impact zones.** The contact point is scaled by the box's half-length and
half-width, which puts each corner at exactly 45°. `zone8` then divides the
outline into `front`, `front_left`, `left`, `rear_left`, `rear`, `rear_right`,
`right`, `front_right`: the central 41 % of each face is the face itself, the
rest belongs to the adjacent corner. `zone4` names the face alone. `*_contact_uv`
keeps the scaled coordinates (u forward, v left) for re-binning.

**Crash types.**

| Type | Struck faces (ego / actor) |
| --- | --- |
| `rear_end_ego_striking` | front / rear |
| `rear_end_ego_struck` | rear / front |
| `head_on` | front / front, headings opposed |
| `angle_front_to_front` | front / front, headings within 90° of each other (e.g. two cars merging nose to nose) |
| `sideswipe_same_direction`, `sideswipe_opposite_direction` | side / side, or front / side with headings within 30° of parallel |
| `angle_ego_striking` | front / side (T-bone, ego into the other) |
| `angle_ego_struck` | side / front (T-bone, the other into the ego) |
| `front_impact`, `side_impact`, `rear_impact` | Pedestrians, animals and objects, whose box has no meaningful front; typed by the ego face alone. |
| `other` | Anything else, e.g. rear / rear. |

The existing collision metric ([collision.py](../src/eval/src/eval/scorers/collision.py))
reports only front / rear / lateral per logged step, on boxes shrunk by 2 %, and
does not say which actor was hit.

## Maneuvers

A maneuver changes the recorded drive from a start time onwards
([maneuvers.py](../src/tools/crash_scenes/maneuvers.py)):

| Kind | What the ego does | Typical crash |
| --- | --- | --- |
| `swerve` | Shifts sideways by 0.75–4.5 m over 2 or 3 s, then holds the shift. | Side-swipe, head-on, parked car, pedestrian at the kerb. |
| `brake` | Brakes to a stop at 3 or 6 m/s² and stays there. | Hit from behind; hit while standing in a crossing path. |
| `hold_speed` | Keeps its speed where the recording slowed down. | Rear-ends the lead vehicle; drives through a crossing. |
| `accelerate` | Speeds up at 2 m/s² by up to 10 m/s. | Rear-ends the lead vehicle. |

Swerves are limited to 6 m/s² lateral acceleration, a 30° heading offset and
speeds above 3 m/s. The search tries each maneuver every 0.5 s of the scene and
keeps, per (actor, struck ego zone, maneuver kind), the gentlest one that works.

A planned crash is kept only if

- the recorded drive does not itself overlap that actor (such overlaps are label
  noise, and those actors are ignored everywhere),
- contact comes at least 0.3 s after the maneuver starts,
- the boxes get at least 0.5 m into each other, so a small tracking error does
  not turn the crash into a miss,
- it happens within the recorded stretch of road.

## How the runs are set up

- **Driver.** `driver=scripted` starts
  [scripted_driver.py](../src/tools/crash_scenes/scripted_driver.py) in place of
  the policy driver. It replays the plan as the reference path; the controller
  and vehicle model track it like any policy output. It runs as its own small
  service rather than as a model plugin, because model plugins are discovered
  from entry points baked into the container image.
- **One plan per scene per run.** The driver is told only the scene id of a
  session. Several crashes in one scene therefore go into separate rounds
  (`round_00`, `round_01`, …), each its own run.
- **No cameras, no renderer, no GPU.** `topology=cpu_headless` launches only the
  driver, the controller and the runtime. Physics is skipped as well: its ground
  correction needs a GPU and adjusts height and tilt, not the path on the map.
- **No in-runtime evaluation.** The built-in evaluator scores camera images and
  filters its summary on them, so `driver=scripted` turns it off. Rollouts are
  labelled with `crash_scenes label` / `confirm` instead, and there is no
  `metrics.parquet`.
- **Same settings as the search.** `tools/run_crash_scenes.sh` passes
  `n_sim_steps`, `force_gt_duration_us`, `control_timestep_us` and
  `min_traffic_duration_us` from `search_config.json`, and the scene catalog
  `scenes.csv` written by `plans`, so the rollout sees the same scene file,
  actors and timing the search did.
- **Early end.** The driver ends each rollout 2 s after the planned crash, or
  after the closest pass for a near miss. A control runs as long as the longest
  crash or near-miss drive of its scene.

## Confirmation

`confirmation.jsonl` has one row per rollout with an `outcome`:

| Variant | Outcomes |
| --- | --- |
| crash | `confirmed` (first crash is with the planned actor), `other_actor`, `no_crash` |
| near_miss | `near_miss` (no crash, target passed within 1.5 m), `crash`, `wide` |
| control | `clean`, `crash` |

A planned drive with no finished rollout gets the outcome `not_run`.

Confirmed crashes also record whether the struck ego zone and crash type match
the plan, and the error in contact time. Every row has `tracking_error_m`, the
largest distance between the driven and the planned ego position.

Use the label of the *rollout*, not the plan, as ground truth for a monitor: the
plan says what was intended, the rollout what happened.

## Layout of a plan set

```
experiments/crash_scenes/plans_<name>/
  selection.jsonl            chosen manifest rows, each with its round
  scenes.csv                 wizard catalog of the scene files used
  search_config.json
  control/plan.json          recorded drive of every scene used
  round_00/crash/plan.json
  round_00/near_miss/plan.json
  runs/<plan name>/          one AlpaSim run directory per plan file
  confirmation.jsonl
```

## Limits

- With no cameras, the first driver call of a rollout comes before the first
  logged actor pose. The two-second monitor raises once on that call (live and
  in replay alike). The call falls inside its warm-up, so no alert is lost;
  `replay-monitor` reports it as `monitor_failures: 1`.

- Other actors never react. That is what makes the crashes cheap to force, and
  it also means the near misses and controls are not evasive on the actors' side.
- Scripted drives have no reasoning text. Testing a reasoning-consistency
  monitor needs a policy in the loop.
- Runs here have no ground correction, so ego height and tilt are not
  meaningful; position and heading on the map are.
- A scene cached in several reconstructions is used in its newest one.
