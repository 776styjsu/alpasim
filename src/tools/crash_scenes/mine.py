# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Search a recorded scene for ego maneuvers that end in a crash.

Actors replay their recording whatever the ego does, so whether a maneuver
crashes is decided by geometry alone: build the ego path, test it against the
recorded actor boxes, and label the first contact. No simulator is involved;
``confirm`` later checks which of these the controller really reproduces.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass

import numpy as np

from .boxes import boxes_overlap
from .label import CrashLabel, find_crashes, gap_to, nearest_pass
from .maneuvers import (
    RECORDED,
    Maneuver,
    RecordedDrive,
    is_drivable,
    plan_arrays,
    progress_at,
    severity,
)
from .scenes import Scene
from .tracks import ActorTrack


@dataclass(frozen=True)
class SearchConfig:
    # Rollout settings the search must share with the confirmation runs.
    n_sim_steps: int = 200
    force_gt_duration_us: int = 500_000
    control_timestep_us: int = 100_000
    min_traffic_duration_us: int = 3_000_000

    # Maneuver grid.
    start_step_us: int = 500_000
    # Earliest maneuver start after the policy takes over, and the time kept
    # free before the rollout ends.
    settle_us: int = 1_000_000
    tail_us: int = 1_000_000
    lateral_shifts_m: tuple[float, ...] = (0.75, 1.5, 2.25, 3.0, 3.75, 4.5)
    ramps_s: tuple[float, ...] = (2.0, 3.0)
    brake_mps2: tuple[float, ...] = (-3.0, -6.0)
    accelerate_mps2: tuple[float, ...] = (2.0,)
    # A speed maneuver must move the ego at least this far from where the
    # recording has it, or it is the recorded drive again.
    min_progress_change_m: float = 2.0

    # What counts as a usable crash.
    min_contact_delay_us: int = 300_000
    # Shallower contacts are corner grazes that tracking error can undo.
    min_penetration_m: float = 0.5

    # Near miss: the same maneuver scaled down until it clears the target.
    near_miss_target_gap_m: float = 0.5
    near_miss_gap_range_m: tuple[float, float] = (0.2, 1.5)

    # The rollout is cut this long after the planned crash or closest pass.
    after_crash_us: int = 2_000_000


def candidate_maneuvers(
    drive: RecordedDrive, scene: Scene, config: SearchConfig
) -> Iterator[Maneuver]:
    """Every drivable maneuver on the search grid for this scene."""
    first_us = scene.window.closed_loop_start_us + config.settle_us
    last_us = scene.window.end_us - config.tail_us
    for start_us in range(first_us, last_us + 1, config.start_step_us):
        maneuvers = [
            Maneuver("swerve", start_us, lateral_m=side * shift, ramp_s=ramp)
            for shift in config.lateral_shifts_m
            for ramp in config.ramps_s
            for side in (1.0, -1.0)
        ]
        maneuvers += [
            Maneuver("brake", start_us, accel_mps2=accel) for accel in config.brake_mps2
        ]
        maneuvers += [
            Maneuver("accelerate", start_us, accel_mps2=accel)
            for accel in config.accelerate_mps2
        ]
        maneuvers.append(Maneuver("hold_speed", start_us))
        for maneuver in maneuvers:
            if is_drivable(drive, maneuver):
                yield maneuver


class SceneSearch:
    """Shared state for testing many maneuvers against one scene."""

    def __init__(self, scene: Scene, config: SearchConfig):
        self.scene = scene
        self.config = config
        self.drive = RecordedDrive.from_trajectory(scene.recording)

        window = scene.window
        in_window = (self.drive.timestamps_us >= window.closed_loop_start_us) & (
            self.drive.timestamps_us <= window.end_us
        )
        self._in_window = in_window
        times = self.drive.timestamps_us[in_window]

        # Actor poses on the drive's clock, for a cheap first pass over every
        # maneuver. Absent actors are parked far away so they never overlap.
        self._actor_sizes = np.array([actor.size for actor in scene.actors])
        self._actor_reach = np.hypot(*self._actor_sizes.T) / 2.0
        self._actor_poses = np.full((len(scene.actors), len(times), 3), 1e9)
        for index, actor in enumerate(scene.actors):
            present = (times >= actor.start_us) & (times <= actor.end_us)
            if present.any():
                self._actor_poses[index, present] = actor.poses_at(times[present])

        # Actors the recorded drive already overlaps are label noise, not
        # something a maneuver caused.
        self.noisy_actor_ids = {
            crash.actor_id
            for crash in self._crashes(self.ego_track(RECORDED), scene.actors)
        }

    def ego_track(self, maneuver: Maneuver) -> ActorTrack:
        positions, yaw = plan_arrays(self.drive, maneuver)
        return self.scene.ego_track(self.drive.timestamps_us, positions, yaw)

    def _crashes(self, ego: ActorTrack, actors: list[ActorTrack]) -> list[CrashLabel]:
        window = self.scene.window
        return find_crashes(ego, actors, window.closed_loop_start_us, window.end_us)

    def _window_poses(self, ego: ActorTrack) -> np.ndarray:
        return np.stack([ego.xy[:, 0], ego.xy[:, 1], ego.yaw], axis=-1)[self._in_window]

    def _actors_near(self, ego: ActorTrack, margin_m: float) -> list[ActorTrack]:
        """Actors whose box could come within ``margin_m`` of the ego's."""
        distance = np.linalg.norm(
            self._actor_poses[:, :, :2] - self._window_poses(ego)[None, :, :2], axis=-1
        )
        reach = float(np.hypot(*ego.size)) / 2.0 + self._actor_reach + margin_m
        near = (distance <= reach[:, None]).any(axis=1)
        return [actor for actor, close in zip(self.scene.actors, near) if close]

    def _touches_anything(self, ego: ActorTrack) -> bool:
        """First pass on the 0.1 s clock: does the ego overlap any actor at all?"""
        return bool(
            boxes_overlap(
                self._window_poses(ego)[None],
                ego.size,
                self._actor_poses,
                self._actor_sizes[:, None],
            ).any()
        )

    def _caused_crashes(self, ego: ActorTrack) -> list[CrashLabel]:
        crashes = self._crashes(ego, self._actors_near(ego, margin_m=1.0))
        return [c for c in crashes if c.actor_id not in self.noisy_actor_ids]

    def crash_of(self, maneuver: Maneuver) -> CrashLabel | None:
        """The crash this maneuver causes, if it is a usable one."""
        config = self.config
        if maneuver.kind != "swerve":
            moved = plan_arrays(self.drive, maneuver)[0] - self.drive.positions
            if (
                np.linalg.norm(moved[:, :2], axis=1).max()
                < config.min_progress_change_m
            ):
                return None  # the recorded drive again

        ego = self.ego_track(maneuver)
        if not self._touches_anything(ego):
            return None
        caused = self._caused_crashes(ego)
        if not caused:
            return None
        crash = caused[0]
        usable = (
            not crash.overlap_at_start
            and crash.time_us >= maneuver.start_us + config.min_contact_delay_us
            and crash.time_us <= self.drive.recorded_end_us
            and progress_at(self.drive, maneuver, crash.time_us)
            <= self.drive.recorded_length_m
            and crash.max_penetration_m >= config.min_penetration_m
        )
        return crash if usable else None

    def near_miss_of(self, maneuver: Maneuver, crash: CrashLabel) -> dict | None:
        """Scale the maneuver down until it passes the crash target closely.

        Bisects the scale for the target gap, then requires the scaled drive to
        touch nothing and to pass the target within the accepted gap range.
        Returns the scale, the gap, and when the closest pass and the end of
        the rollout fall.
        """
        config = self.config
        window = self.scene.window
        target = next(a for a in self.scene.actors if a.actor_id == crash.actor_id)

        def target_gap(scale: float) -> float:
            return gap_to(
                self.ego_track(maneuver.scaled(scale)),
                target,
                window.closed_loop_start_us,
                window.end_us,
            )

        clear, touching = 0.0, 1.0
        for _ in range(12):
            middle = (clear + touching) / 2.0
            if target_gap(middle) > config.near_miss_target_gap_m:
                clear = middle
            else:
                touching = middle
        if clear == 0.0:
            return None  # the recorded drive itself already passes this close

        ego = self.ego_track(maneuver.scaled(clear))
        gap, time_us = nearest_pass(
            ego, target, window.closed_loop_start_us, window.end_us
        )
        low, high = config.near_miss_gap_range_m
        if not low <= gap <= high:
            return None
        if self._caused_crashes(ego):
            return None
        return {
            "scale": round(clear, 4),
            "target_gap_m": round(gap, 3),
            "time_us": time_us,
            # A scaled-down maneuver reaches its closest pass later than the
            # crash it was derived from, so it gets its own end time.
            "end_us": min(time_us + config.after_crash_us, window.end_us),
        }


def mine_scene(scene: Scene, config: SearchConfig) -> list[dict]:
    """Planned crashes for one scene: one per (actor, struck ego zone, maneuver kind).

    Of the maneuvers that reach the same outcome, the gentlest is kept.
    """
    if not scene.actors:
        return []
    search = SceneSearch(scene, config)

    best: dict[tuple[str, str, str], tuple[Maneuver, CrashLabel]] = {}
    for maneuver in candidate_maneuvers(search.drive, scene, config):
        crash = search.crash_of(maneuver)
        if crash is None:
            continue
        key = (crash.actor_id, crash.ego_zone8, maneuver.kind)
        kept = best.get(key)
        if kept is None or severity(maneuver) < severity(kept[0]):
            best[key] = (maneuver, crash)

    rows = []
    for (actor_id, ego_zone8, kind), (maneuver, crash) in sorted(best.items()):
        rows.append(
            {
                "crash_id": f"{scene.scene_id}|{actor_id.split('@')[0]}|{ego_zone8}|{kind}",
                "scene_id": scene.scene_id,
                "usdz_path": scene.usdz_path,
                "maneuver": asdict(maneuver),
                "predicted": asdict(crash),
                "near_miss": search.near_miss_of(maneuver, crash),
                "end_us": min(
                    crash.time_us + config.after_crash_us, scene.window.end_us
                ),
                "window": asdict(scene.window),
            }
        )
    return rows
