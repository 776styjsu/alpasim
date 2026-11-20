# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Load a recorded scene the way a headless AlpaSim rollout sees it."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
from alpasim_runtime.unbound_rollout import get_ds_rig_to_aabb_center_transform
from alpasim_utils.artifact import Artifact
from alpasim_utils.geometry import Trajectory
from alpasim_utils.scenario import VehicleConfig

from .tracks import EGO_ID, ActorTrack, make_track


def _version_key(version_string: str) -> tuple[int, ...]:
    """Sort key for reconstruction versions such as ``26.1.112-59673f88``."""
    return tuple(int(part) for part in version_string.split("-")[0].split("."))


def index_scenes(usdz_dir: str) -> dict[str, Artifact]:
    """Scene id -> artifact, for every scene file in ``usdz_dir``.

    A scene can be cached in several reconstructions; the newest one is used.
    """
    newest: dict[str, Artifact] = {}
    for path in sorted(glob.glob(os.path.join(usdz_dir, "*.usdz"))):
        artifact = Artifact(path)
        current = newest.get(artifact.scene_id)
        if current is None or _version_key(
            artifact.metadata.version_string
        ) > _version_key(current.metadata.version_string):
            newest[artifact.scene_id] = artifact
    return newest


def write_scenes_csv(artifacts: list[Artifact], csv_path: str) -> None:
    """Write a wizard scene catalog naming exactly these local scene files.

    Passing it as ``scenes.scenes_csv`` makes the wizard run the same
    reconstruction of each scene that was indexed here.
    """
    pl.DataFrame(
        [
            {
                "uuid": str(artifact.metadata.uuid),
                "scene_id": artifact.scene_id,
                "nre_version_string": artifact.metadata.version_string,
                "path": os.path.abspath(artifact.source),
                "last_modified": datetime.fromtimestamp(
                    os.path.getmtime(artifact.source)
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "artifact_repository": "local",
                "hf_revision": "",
            }
            for artifact in artifacts
        ]
    ).write_csv(csv_path)


@dataclass(frozen=True)
class RolloutWindow:
    """Timing of a rollout with no cameras, mirroring the runtime's own."""

    start_us: int
    closed_loop_start_us: int
    end_us: int

    @staticmethod
    def for_recording(
        recording: Trajectory,
        n_sim_steps: int,
        force_gt_duration_us: int,
        control_timestep_us: int,
    ) -> RolloutWindow:
        start_us = recording.time_range_us.start
        last_recorded_us = recording.time_range_us.stop - 1
        steps = min(n_sim_steps, (last_recorded_us - start_us) // control_timestep_us)
        return RolloutWindow(
            start_us=start_us,
            closed_loop_start_us=start_us + force_gt_duration_us,
            end_us=start_us + steps * control_timestep_us,
        )


@dataclass(frozen=True)
class Scene:
    scene_id: str
    usdz_path: str
    window: RolloutWindow
    # Recorded pose of the ego rig (rear axle, on the ground) in the scene frame.
    recording: Trajectory
    vehicle: VehicleConfig
    actors: list[ActorTrack]

    def ego_track(
        self, timestamps_us: np.ndarray, rig_positions: np.ndarray, rig_yaw: np.ndarray
    ) -> ActorTrack:
        """The ego's box track when its rig follows the given planar path."""
        forward, left, _ = get_ds_rig_to_aabb_center_transform(self.vehicle).vec3
        cos, sin = np.cos(rig_yaw), np.sin(rig_yaw)
        centre = np.stack(
            [
                rig_positions[:, 0] + forward * cos - left * sin,
                rig_positions[:, 1] + forward * sin + left * cos,
            ],
            axis=-1,
        )
        return make_track(
            actor_id=EGO_ID,
            label="",
            length_m=self.vehicle.aabb_x_m,
            width_m=self.vehicle.aabb_y_m,
            timestamps_us=timestamps_us,
            xy=centre,
            yaw=rig_yaw,
        )


def load_scene(
    artifact: Artifact,
    n_sim_steps: int,
    force_gt_duration_us: int,
    control_timestep_us: int,
    min_traffic_duration_us: int,
) -> Scene:
    """Recorded ego path plus the actors a rollout of this scene would contain.

    Actors are clipped to the rollout window and dropped when they appear for
    less than ``min_traffic_duration_us``, as ``UnboundRollout.create`` does.
    """
    rig = artifact.rig
    if rig.vehicle_config is None:
        raise ValueError(f"{artifact.scene_id}: no vehicle config in the scene")

    window = RolloutWindow.for_recording(
        rig.trajectory, n_sim_steps, force_gt_duration_us, control_timestep_us
    )
    traffic = artifact.traffic_objects.clip_trajectories(
        window.start_us, window.end_us + 1, exclude_empty=True
    ).filter_short_trajectories(min_traffic_duration_us)

    actors = [
        make_track(
            actor_id=obj.track_id,
            label=obj.label_class,
            length_m=obj.aabb.x,
            width_m=obj.aabb.y,
            timestamps_us=obj.trajectory.timestamps_us,
            xy=obj.trajectory.positions[:, :2],
            yaw=obj.trajectory.yaws,
            is_static=obj.is_static,
        )
        for obj in traffic.values()
    ]
    return Scene(
        scene_id=artifact.scene_id,
        usdz_path=artifact.source,
        window=window,
        recording=rig.trajectory,
        vehicle=rig.vehicle_config,
        actors=[actor for actor in actors if len(actor.timestamps_us) >= 2],
    )
