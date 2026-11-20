# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Choose mined crashes and write the plan files the scripted driver follows.

A plan file holds one ego path per scene, because the driver is told only the
scene id of a session. Several crashes in the same scene therefore go into
separate rounds, each of which is its own AlpaSim run.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from alpasim_utils.artifact import Artifact

from .maneuvers import RECORDED, Maneuver, RecordedDrive, plan_arrays
from .scenes import write_scenes_csv

VARIANTS = ("crash", "near_miss", "control")


@dataclass(frozen=True)
class Plan:
    """The rig path the ego follows in one scene, in the scene frame."""

    scene_id: str
    variant: str
    # Crash this plan belongs to; None for the recorded drive of a whole scene.
    crash_id: str | None
    # The driver ends the rollout at this time.
    end_us: int
    timestamps_us: np.ndarray  # (N,) int64
    positions: np.ndarray  # (N, 3)
    yaw: np.ndarray  # (N,) unwrapped

    def to_json(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "variant": self.variant,
            "crash_id": self.crash_id,
            "end_us": self.end_us,
            "timestamps_us": self.timestamps_us.tolist(),
            "positions": np.round(self.positions, 4).tolist(),
            "yaw": np.round(self.yaw, 6).tolist(),
        }

    @staticmethod
    def from_json(data: dict) -> Plan:
        return Plan(
            scene_id=data["scene_id"],
            variant=data["variant"],
            crash_id=data["crash_id"],
            end_us=data["end_us"],
            timestamps_us=np.array(data["timestamps_us"], dtype=np.int64),
            positions=np.array(data["positions"], dtype=float),
            yaw=np.array(data["yaw"], dtype=float),
        )


def write_plan_file(plans: list[Plan], path: Path) -> None:
    scene_ids = [plan.scene_id for plan in plans]
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError(f"{path}: more than one plan for a scene")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"plans": [plan.to_json() for plan in plans]}))


def read_plan_file(path: str) -> dict[str, Plan]:
    """Plans keyed by scene id."""
    plans = [
        Plan.from_json(data) for data in json.loads(Path(path).read_text())["plans"]
    ]
    return {plan.scene_id: plan for plan in plans}


def select_crashes(rows: list[dict], per_cell: int, max_per_scene: int) -> list[dict]:
    """Pick a spread of crashes from a manifest.

    A cell is (actor group, struck ego face, crash type). Cells are filled
    rarest first, up to ``per_cell`` crashes each, preferring crashes that have
    a near-miss variant and scenes that have been used least, and never taking
    more than ``max_per_scene`` crashes from one scene.
    """
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        predicted = row["predicted"]
        cells[
            (predicted["actor_group"], predicted["ego_zone4"], predicted["crash_type"])
        ].append(row)

    used: dict[str, int] = defaultdict(int)
    selected: list[dict] = []
    for cell in sorted(cells, key=lambda key: (len(cells[key]), key)):
        taken = 0
        candidates = sorted(cells[cell], key=lambda row: row["crash_id"])
        while taken < per_cell:
            open_rows = [
                row for row in candidates if used[row["scene_id"]] < max_per_scene
            ]
            if not open_rows:
                break
            row = min(
                open_rows,
                key=lambda row: (row["near_miss"] is None, used[row["scene_id"]]),
            )
            candidates.remove(row)
            used[row["scene_id"]] += 1
            selected.append(row)
            taken += 1
    return selected


def write_plan_set(selected: list[dict], output_dir: Path) -> dict:
    """Write plan files, a scene catalog and the selection for a set of crashes.

    Layout under ``output_dir``::

        selection.jsonl          the chosen manifest rows, each with its round
        scenes.csv               wizard catalog of the scene files used
        control/plan.json        the recorded drive of every scene used
        round_00/crash/plan.json
        round_00/near_miss/plan.json
        ...
    """
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(selected, key=lambda row: row["crash_id"]):
        by_scene[row["scene_id"]].append(row)

    rounds: dict[int, dict[str, list[Plan]]] = defaultdict(lambda: defaultdict(list))
    control: list[Plan] = []
    artifacts: list[Artifact] = []
    selection_rows: list[dict] = []

    for scene_id, rows in sorted(by_scene.items()):
        artifact = Artifact(rows[0]["usdz_path"])
        artifacts.append(artifact)
        drive = RecordedDrive.from_trajectory(artifact.rig.trajectory)

        def plan(maneuver: Maneuver, variant: str, crash_id: str | None, end_us: int):
            positions, yaw = plan_arrays(drive, maneuver)
            return Plan(
                scene_id=scene_id,
                variant=variant,
                crash_id=crash_id,
                end_us=end_us,
                timestamps_us=drive.timestamps_us,
                positions=positions,
                yaw=yaw,
            )

        # The control covers the longest of the scene's crash and near-miss drives.
        control_end_us = 0
        for round_index, row in enumerate(rows):
            maneuver = Maneuver(**row["maneuver"])
            crash_id = row["crash_id"]
            rounds[round_index]["crash"].append(
                plan(maneuver, "crash", crash_id, row["end_us"])
            )
            control_end_us = max(control_end_us, row["end_us"])
            near_miss = row["near_miss"]
            if near_miss is not None:
                rounds[round_index]["near_miss"].append(
                    plan(
                        maneuver.scaled(near_miss["scale"]),
                        "near_miss",
                        crash_id,
                        near_miss["end_us"],
                    )
                )
                control_end_us = max(control_end_us, near_miss["end_us"])
            selection_rows.append({**row, "round": round_index})
        control.append(plan(RECORDED, "control", None, control_end_us))

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selection.jsonl").open("w") as out:
        for row in selection_rows:
            out.write(json.dumps(row) + "\n")
    write_scenes_csv(artifacts, str(output_dir / "scenes.csv"))
    write_plan_file(control, output_dir / "control" / "plan.json")
    for round_index, variants in sorted(rounds.items()):
        for variant, plans in variants.items():
            write_plan_file(
                plans, output_dir / f"round_{round_index:02d}" / variant / "plan.json"
            )

    return {
        "crashes": len(selection_rows),
        "scenes": len(by_scene),
        "rounds": len(rounds),
        "near_misses": sum(row["near_miss"] is not None for row in selection_rows),
    }
