# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Actor box tracks, and reading them from a ``rollout.asl``."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from alpasim_utils.logs import async_read_pb_log

EGO_ID = "EGO"


@dataclass(frozen=True)
class ActorTrack:
    """One actor's box over time: centre position, heading and size."""

    actor_id: str
    label: str
    length_m: float
    width_m: float
    timestamps_us: np.ndarray  # (N,) int64, strictly increasing
    xy: np.ndarray  # (N, 2) box centre
    yaw: np.ndarray  # (N,) unwrapped heading, radians
    is_static: bool = False

    @property
    def size(self) -> np.ndarray:
        return np.array([self.length_m, self.width_m])

    @property
    def start_us(self) -> int:
        return int(self.timestamps_us[0])

    @property
    def end_us(self) -> int:
        return int(self.timestamps_us[-1])

    def poses_at(self, times_us: np.ndarray) -> np.ndarray:
        """Linearly interpolated ``(x, y, yaw)`` at times inside the track."""
        times = np.asarray(times_us, dtype=np.int64)
        if times.size and (times.min() < self.start_us or times.max() > self.end_us):
            raise ValueError(
                f"{self.actor_id}: times {times.min()}..{times.max()} leave the "
                f"track {self.start_us}..{self.end_us}"
            )
        return np.stack(
            [
                np.interp(times, self.timestamps_us, self.xy[:, 0]),
                np.interp(times, self.timestamps_us, self.xy[:, 1]),
                np.interp(times, self.timestamps_us, self.yaw),
            ],
            axis=-1,
        )

    def velocity_at(self, time_us: int, window_us: int = 200_000) -> np.ndarray:
        """Velocity over the ``window_us`` leading up to ``time_us``, in m/s."""
        end = min(max(time_us, self.start_us), self.end_us)
        begin = max(end - window_us, self.start_us)
        if end == begin:
            end = min(begin + window_us, self.end_us)
        if end == begin:
            return np.zeros(2)
        first, last = self.poses_at(np.array([begin, end]))
        return (last[:2] - first[:2]) / ((end - begin) / 1e6)


def make_track(
    actor_id: str,
    label: str,
    length_m: float,
    width_m: float,
    timestamps_us: np.ndarray,
    xy: np.ndarray,
    yaw: np.ndarray,
    is_static: bool = False,
) -> ActorTrack:
    """Build a track from samples in any order; later duplicates of a timestamp win."""
    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    # Reverse before the stable unique so the last sample of each timestamp is kept.
    unique, first_in_reversed = np.unique(timestamps[::-1], return_index=True)
    keep = len(timestamps) - 1 - first_in_reversed
    return ActorTrack(
        actor_id=actor_id,
        label=label,
        length_m=float(length_m),
        width_m=float(width_m),
        timestamps_us=unique,
        xy=np.asarray(xy, dtype=float)[keep],
        yaw=np.unwrap(np.asarray(yaw, dtype=float)[keep]),
        is_static=is_static,
    )


@dataclass(frozen=True)
class RolloutTracks:
    """Everything the labeller needs from one rollout."""

    scene_id: str
    session_uuid: str
    ego: ActorTrack
    actors: dict[str, ActorTrack]
    # First timestamp at which the policy, not the recording, drives the ego.
    closed_loop_start_us: int


def _yaw(quat) -> float:
    return math.atan2(
        2.0 * (quat.w * quat.z + quat.x * quat.y),
        1.0 - 2.0 * (quat.y**2 + quat.z**2),
    )


async def read_rollout_tracks(asl_path: str) -> RolloutTracks:
    """Read box tracks, labels and the closed-loop start from a ``rollout.asl``.

    Logged poses are box centres for every actor, the ego included.
    """
    metadata = None
    static_ids: set[str] = set()
    samples: dict[str, list[tuple[int, float, float, float]]] = {}

    async for entry in async_read_pb_log(asl_path, raise_on_malformed=True):
        kind = entry.WhichOneof("log_entry")
        if kind == "rollout_metadata":
            metadata = entry.rollout_metadata
        elif kind == "traffic_session_request":
            static_ids = {
                obj.object_id
                for obj in entry.traffic_session_request.logged_object_trajectories
                if obj.is_static
            }
        elif kind == "actor_poses":
            timestamp_us = int(entry.actor_poses.timestamp_us)
            for actor in entry.actor_poses.actor_poses:
                pose = actor.actor_pose
                samples.setdefault(actor.actor_id, []).append(
                    (timestamp_us, pose.vec.x, pose.vec.y, _yaw(pose.quat))
                )

    if metadata is None:
        raise ValueError(f"No rollout metadata in {asl_path}")

    tracks: dict[str, ActorTrack] = {}
    for definition in metadata.actor_definitions.actor_aabb:
        actor_samples = samples.get(definition.actor_id)
        if actor_samples is None:
            continue  # defined for the scene but never present in this rollout
        data = np.array(actor_samples, dtype=float)
        tracks[definition.actor_id] = make_track(
            actor_id=definition.actor_id,
            label=definition.actor_label,
            length_m=definition.aabb.size_x,
            width_m=definition.aabb.size_y,
            timestamps_us=data[:, 0].astype(np.int64),
            xy=data[:, 1:3],
            yaw=data[:, 3],
            is_static=definition.actor_id in static_ids,
        )

    session = metadata.session_metadata
    return RolloutTracks(
        scene_id=session.scene_id,
        session_uuid=session.session_uuid,
        ego=tracks.pop(EGO_ID),
        actors=tracks,
        closed_loop_start_us=int(session.render_start_timestamp_us)
        + int(metadata.force_gt_duration),
    )
