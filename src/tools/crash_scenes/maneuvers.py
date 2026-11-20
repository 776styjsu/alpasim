# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Ego maneuvers that depart from the recorded drive.

A maneuver changes the recorded drive along two independent channels: how far
along the recorded path the ego is at each moment (braking, not braking,
accelerating), and how far it sits to the side of that path (a swerve). The
resulting rig path is what the scripted driver hands to the controller.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from alpasim_utils.geometry import Trajectory

# Peak lateral speed and acceleration of the quintic ramp, as multiples of
# shift / ramp time and shift / ramp time squared.
_RAMP_PEAK_SPEED = 1.875
_RAMP_PEAK_ACCEL = 5.774

# Limits a swerve must respect to be something a car can do.
MAX_LATERAL_ACCEL_MPS2 = 6.0
MAX_HEADING_OFFSET_RAD = np.radians(30.0)
MIN_SWERVE_SPEED_MPS = 3.0
# Below this speed the direction of travel no longer defines a heading.
MIN_HEADING_SPEED_MPS = 0.5
MAX_SPEED_MPS = 35.0
ACCELERATE_SPEED_GAIN_MPS = 10.0
# How long a plan runs past the end of the recording, so the controller keeps
# a full look-ahead up to the last simulated step.
PLAN_EXTENSION_US = 5_000_000
_STEP_US = 100_000


@dataclass(frozen=True)
class Maneuver:
    """One departure from the recorded drive, starting at ``start_us``.

    ``kind`` is ``none`` (the recorded drive), ``swerve``, ``brake``,
    ``hold_speed`` or ``accelerate``. ``scale`` blends from the recorded drive
    (0) to the full maneuver (1); a near miss is a crash maneuver scaled down
    until it clears.
    """

    kind: str
    start_us: int = 0
    lateral_m: float = 0.0  # swerve: final shift, positive to the left
    ramp_s: float = 0.0  # swerve: time taken to reach the shift
    accel_mps2: float = 0.0  # brake (negative) or accelerate (positive)
    scale: float = 1.0

    def scaled(self, scale: float) -> Maneuver:
        return replace(self, scale=scale)


RECORDED = Maneuver(kind="none", scale=0.0)


@dataclass(frozen=True)
class RecordedDrive:
    """The recorded rig path on a uniform clock, extended straight past its end."""

    timestamps_us: np.ndarray  # (N,) int64
    positions: np.ndarray  # (N, 3)
    yaw: np.ndarray  # (N,) unwrapped
    progress_m: np.ndarray  # (N,) distance travelled along the path, increasing
    speed_mps: np.ndarray  # (N,)
    recorded_end_us: int
    recorded_length_m: float

    @staticmethod
    def from_trajectory(recording: Trajectory) -> RecordedDrive:
        start_us = recording.time_range_us.start
        end_us = recording.time_range_us.stop - 1
        recorded_us = np.arange(start_us, end_us + 1, _STEP_US, dtype=np.int64)
        sampled = recording.interpolate(recorded_us.astype(np.uint64))
        positions = sampled.positions.astype(float)
        yaw = np.unwrap(sampled.yaws.astype(float))

        steps = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
        progress = np.concatenate([[0.0], np.cumsum(steps)])
        speed = np.gradient(progress, recorded_us / 1e6)

        # Past the recording the drive continues straight at its final speed.
        extra_us = np.arange(_STEP_US, PLAN_EXTENSION_US + 1, _STEP_US, dtype=np.int64)
        extra_m = speed[-1] * extra_us / 1e6
        heading = np.array([np.cos(yaw[-1]), np.sin(yaw[-1]), 0.0])
        return RecordedDrive(
            timestamps_us=np.concatenate([recorded_us, recorded_us[-1] + extra_us]),
            positions=np.concatenate(
                [positions, positions[-1] + extra_m[:, None] * heading]
            ),
            yaw=np.concatenate([yaw, np.full(len(extra_us), yaw[-1])]),
            progress_m=np.concatenate([progress, progress[-1] + extra_m]),
            speed_mps=np.concatenate([speed, np.full(len(extra_us), speed[-1])]),
            recorded_end_us=int(recorded_us[-1]),
            recorded_length_m=float(progress[-1]),
        )

    def speed_at(self, time_us: int) -> float:
        return float(np.interp(time_us, self.timestamps_us, self.speed_mps))


def _progress(drive: RecordedDrive, maneuver: Maneuver) -> np.ndarray:
    """Distance along the recorded path at each timestamp, for the full maneuver."""
    if maneuver.kind in ("none", "swerve"):
        return drive.progress_m

    elapsed = np.clip((drive.timestamps_us - maneuver.start_us) / 1e6, 0.0, None)
    start_m = float(np.interp(maneuver.start_us, drive.timestamps_us, drive.progress_m))
    speed = drive.speed_at(maneuver.start_us)

    if maneuver.kind == "hold_speed":
        travelled = speed * elapsed
    elif maneuver.kind == "brake":
        until_stopped = np.minimum(elapsed, speed / -maneuver.accel_mps2)
        travelled = speed * until_stopped + 0.5 * maneuver.accel_mps2 * until_stopped**2
    elif maneuver.kind == "accelerate":
        top_speed = min(speed + ACCELERATE_SPEED_GAIN_MPS, MAX_SPEED_MPS)
        speeding_up = np.minimum(elapsed, (top_speed - speed) / maneuver.accel_mps2)
        travelled = (
            speed * speeding_up
            + 0.5 * maneuver.accel_mps2 * speeding_up**2
            + top_speed * (elapsed - speeding_up)
        )
    else:
        raise ValueError(f"Unknown maneuver kind {maneuver.kind!r}")

    return np.where(
        drive.timestamps_us < maneuver.start_us, drive.progress_m, start_m + travelled
    )


def _lateral(drive: RecordedDrive, maneuver: Maneuver) -> np.ndarray:
    """Sideways shift from the recorded path at each timestamp, for the full maneuver."""
    if maneuver.kind != "swerve":
        return np.zeros(len(drive.timestamps_us))
    phase = np.clip(
        (drive.timestamps_us - maneuver.start_us) / 1e6 / maneuver.ramp_s, 0.0, 1.0
    )
    return maneuver.lateral_m * (10 * phase**3 - 15 * phase**4 + 6 * phase**5)


def severity(maneuver: Maneuver) -> float:
    """Peak acceleration the maneuver asks for, in m/s^2."""
    if maneuver.kind == "swerve":
        return _RAMP_PEAK_ACCEL * abs(maneuver.lateral_m) / maneuver.ramp_s**2
    return abs(maneuver.accel_mps2)


def is_drivable(drive: RecordedDrive, maneuver: Maneuver) -> bool:
    """Whether the maneuver stays within what a car can physically do."""
    speed = drive.speed_at(maneuver.start_us)
    if maneuver.kind == "swerve":
        ramp = (drive.timestamps_us >= maneuver.start_us) & (
            drive.timestamps_us <= maneuver.start_us + maneuver.ramp_s * 1e6
        )
        slowest = float(drive.speed_mps[ramp].min())
        if slowest < MIN_SWERVE_SPEED_MPS:
            return False
        peak_lateral_speed = (
            _RAMP_PEAK_SPEED * abs(maneuver.lateral_m) / maneuver.ramp_s
        )
        return (
            severity(maneuver) <= MAX_LATERAL_ACCEL_MPS2
            and np.arctan2(peak_lateral_speed, slowest) <= MAX_HEADING_OFFSET_RAD
        )
    if maneuver.kind == "brake":
        return speed > 1.0
    if maneuver.kind == "accelerate":
        return speed < MAX_SPEED_MPS - 1.0
    return True


def plan_arrays(
    drive: RecordedDrive, maneuver: Maneuver
) -> tuple[np.ndarray, np.ndarray]:
    """Rig positions ``(N, 3)`` and headings ``(N,)`` under ``maneuver``.

    Sampled at ``drive.timestamps_us``, in the scene frame.
    """
    progress = (1.0 - maneuver.scale) * drive.progress_m + maneuver.scale * _progress(
        drive, maneuver
    )
    lateral = maneuver.scale * _lateral(drive, maneuver)

    # ``progress_m`` is flat while the ego stands still; a vanishing slope keeps
    # the lookup well defined there.
    lookup = drive.progress_m + 1e-9 * np.arange(len(drive.progress_m))
    positions = np.stack(
        [np.interp(progress, lookup, drive.positions[:, axis]) for axis in range(3)],
        axis=-1,
    )
    path_yaw = np.interp(progress, lookup, drive.yaw)

    seconds = drive.timestamps_us / 1e6
    forward_speed = np.gradient(progress, seconds)
    lateral_speed = np.gradient(lateral, seconds)
    yaw = path_yaw + np.where(
        forward_speed > MIN_HEADING_SPEED_MPS,
        np.arctan2(lateral_speed, np.maximum(forward_speed, MIN_HEADING_SPEED_MPS)),
        0.0,
    )

    positions[:, 0] -= lateral * np.sin(path_yaw)
    positions[:, 1] += lateral * np.cos(path_yaw)
    return positions, yaw


def progress_at(drive: RecordedDrive, maneuver: Maneuver, time_us: int) -> float:
    """Distance along the recorded path the ego has covered at ``time_us``."""
    progress = (1.0 - maneuver.scale) * drive.progress_m + maneuver.scale * _progress(
        drive, maneuver
    )
    return float(np.interp(time_us, drive.timestamps_us, progress))
