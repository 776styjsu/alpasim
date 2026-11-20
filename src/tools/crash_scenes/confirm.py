# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Check which planned drives came out as planned in AlpaSim.

The search assumes the ego follows its plan exactly. In a rollout the
controller and the vehicle model track the plan with some error, so a planned
crash can miss and a planned near miss can touch. This compares the label of
each rollout with what its plan predicted.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from alpasim_utils.artifact import Artifact

from .label import gap_to, label_tracks
from .mine import SceneSearch, SearchConfig
from .plans import Plan, read_plan_file
from .scenes import Scene, load_scene
from .tracks import RolloutTracks, read_rollout_tracks

PLAN_FILENAME = "plan.json"
# A near miss must pass its target closer than this to count as one.
NEAR_MISS_MAX_GAP_M = 1.5


def _tracking_error_m(
    rollout: RolloutTracks, scene: Scene, plan: Plan, until_us: int
) -> float:
    """Largest distance between the driven and the planned ego box centre."""
    planned = scene.ego_track(plan.timestamps_us, plan.positions, plan.yaw)
    times = rollout.ego.timestamps_us
    times = times[(times >= rollout.closed_loop_start_us) & (times <= until_us)]
    if times.size == 0:
        return 0.0
    offset = rollout.ego.poses_at(times)[:, :2] - planned.poses_at(times)[:, :2]
    return float(np.linalg.norm(offset, axis=1).max())


def confirm_rollout(
    rollout: RolloutTracks,
    plan: Plan,
    selection_row: dict | None,
    scene: Scene,
    noisy_actor_ids: set[str],
) -> dict:
    """Compare one rollout with its plan.

    ``outcome`` is, by plan variant:

    * crash: ``confirmed`` (first crash is with the planned actor),
      ``other_actor`` or ``no_crash``;
    * near_miss: ``near_miss`` (no crash, target passed within
      ``NEAR_MISS_MAX_GAP_M``), ``crash`` or ``wide``;
    * control: ``clean`` or ``crash``.

    Actors the recorded drive already overlaps are ignored throughout, as in
    the search.
    """
    crashes, _ = label_tracks(
        rollout.ego,
        [a for a in rollout.actors.values() if a.actor_id not in noisy_actor_ids],
        rollout.closed_loop_start_us,
        rollout.ego.end_us,
    )
    first = crashes[0] if crashes else None
    row = {
        "scene_id": rollout.scene_id,
        "session_uuid": rollout.session_uuid,
        "variant": plan.variant,
        "crash_id": plan.crash_id,
        "crash": None if first is None else asdict(first),
        "tracking_error_m": round(
            _tracking_error_m(
                rollout,
                scene,
                plan,
                until_us=rollout.ego.end_us if first is None else first.time_us,
            ),
            3,
        ),
    }

    if plan.variant == "control":
        row["outcome"] = "clean" if first is None else "crash"
        return row

    assert selection_row is not None
    predicted = selection_row["predicted"]
    target = rollout.actors[predicted["actor_id"]]
    row["target_gap_m"] = round(
        gap_to(rollout.ego, target, rollout.closed_loop_start_us, rollout.ego.end_us), 3
    )

    if plan.variant == "near_miss":
        if first is not None:
            row["outcome"] = "crash"
        else:
            row["outcome"] = (
                "near_miss" if row["target_gap_m"] <= NEAR_MISS_MAX_GAP_M else "wide"
            )
        return row

    if first is None:
        row["outcome"] = "no_crash"
    elif first.actor_id != predicted["actor_id"]:
        row["outcome"] = "other_actor"
    else:
        row["outcome"] = "confirmed"
        row["same_ego_zone8"] = first.ego_zone8 == predicted["ego_zone8"]
        row["same_ego_zone4"] = first.ego_zone4 == predicted["ego_zone4"]
        row["same_crash_type"] = first.crash_type == predicted["crash_type"]
        row["contact_time_error_s"] = round(
            (first.time_us - predicted["time_us"]) / 1e6, 3
        )
    return row


def confirm_runs(plan_set_dir: Path, run_dirs: list[Path]) -> list[dict]:
    """Confirm every finished rollout under ``run_dirs`` against the plan it ran.

    Each run directory holds the ``plan.json`` its driver followed. Rollouts
    without the runtime's ``_complete`` marker failed or were interrupted and
    are left out.
    """
    selection = {
        row["crash_id"]: row
        for row in map(json.loads, (plan_set_dir / "selection.jsonl").open())
    }
    config = SearchConfig(
        **json.loads((plan_set_dir / "search_config.json").read_text())
    )
    usdz_paths = {row["scene_id"]: row["usdz_path"] for row in selection.values()}

    scenes: dict[str, tuple[Scene, set[str]]] = {}
    results = []
    for run_dir in run_dirs:
        plans = read_plan_file(str(run_dir / PLAN_FILENAME))
        for asl_path in sorted(run_dir.rglob("rollout.asl")):
            if not (asl_path.parent / "_complete").exists():
                continue
            rollout = asyncio.run(read_rollout_tracks(str(asl_path)))
            if rollout.scene_id not in scenes:
                scene = load_scene(
                    Artifact(usdz_paths[rollout.scene_id]),
                    config.n_sim_steps,
                    config.force_gt_duration_us,
                    config.control_timestep_us,
                    config.min_traffic_duration_us,
                )
                scenes[rollout.scene_id] = (
                    scene,
                    SceneSearch(scene, config).noisy_actor_ids,
                )
            scene, noisy_actor_ids = scenes[rollout.scene_id]
            plan = plans[rollout.scene_id]
            row = confirm_rollout(
                rollout,
                plan,
                selection.get(plan.crash_id),
                scene,
                noisy_actor_ids,
            )
            row["asl_path"] = str(asl_path)
            results.append(row)
            del plans[rollout.scene_id]
        # Every planned drive gets a row, also those with no finished rollout.
        results.extend(
            {
                "scene_id": plan.scene_id,
                "variant": plan.variant,
                "crash_id": plan.crash_id,
                "outcome": "not_run",
            }
            for plan in plans.values()
        )
    return results
