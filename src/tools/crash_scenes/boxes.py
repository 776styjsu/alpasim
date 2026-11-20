# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Planar oriented-box geometry: overlap, penetration, gap and impact zones.

A box is its centre pose ``(x, y, yaw)`` plus ``(length, width)``; length runs
along the heading. Pose arrays broadcast, so one call covers every timestep of
a track.
"""

from __future__ import annotations

import math

import numpy as np

# Counter-clockwise from the front, 45 degrees apart.
ZONES8 = (
    "front",
    "front_left",
    "left",
    "rear_left",
    "rear",
    "rear_right",
    "right",
    "front_right",
)
ZONES4 = ("front", "left", "rear", "right")


def box_corners(pose: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Corners of boxes with poses ``(..., 3)`` and sizes ``(..., 2)``.

    Returns ``(..., 4, 2)`` in the order front-left, rear-left, rear-right,
    front-right (counter-clockwise).
    """
    pose = np.asarray(pose, dtype=float)
    half = np.asarray(size, dtype=float) / 2.0
    cos, sin = np.cos(pose[..., 2]), np.sin(pose[..., 2])
    signs = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    local = signs * half[..., None, :]
    x = (
        pose[..., None, 0]
        + local[..., 0] * cos[..., None]
        - local[..., 1] * sin[..., None]
    )
    y = (
        pose[..., None, 1]
        + local[..., 0] * sin[..., None]
        + local[..., 1] * cos[..., None]
    )
    return np.stack([x, y], axis=-1)


def _separation_slack(
    pose_a: np.ndarray, size_a: np.ndarray, pose_b: np.ndarray, size_b: np.ndarray
) -> np.ndarray:
    """Smallest overlap of boxes A and B along any of their four face normals.

    Negative when some normal separates the boxes (separating-axis test).
    """
    pose_a = np.asarray(pose_a, dtype=float)
    pose_b = np.asarray(pose_b, dtype=float)
    half_a = np.asarray(size_a, dtype=float) / 2.0
    half_b = np.asarray(size_b, dtype=float) / 2.0
    delta = pose_b[..., :2] - pose_a[..., :2]

    def axes(yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cos, sin = np.cos(yaw), np.sin(yaw)
        return np.stack([cos, sin], axis=-1), np.stack([-sin, cos], axis=-1)

    a_long, a_lat = axes(pose_a[..., 2])
    b_long, b_lat = axes(pose_b[..., 2])

    slack = np.full(np.broadcast_shapes(pose_a.shape[:-1], pose_b.shape[:-1]), np.inf)
    for axis in (a_long, a_lat, b_long, b_lat):
        reach_a = half_a[..., 0] * np.abs((a_long * axis).sum(-1)) + half_a[
            ..., 1
        ] * np.abs((a_lat * axis).sum(-1))
        reach_b = half_b[..., 0] * np.abs((b_long * axis).sum(-1)) + half_b[
            ..., 1
        ] * np.abs((b_lat * axis).sum(-1))
        slack = np.minimum(slack, reach_a + reach_b - np.abs((delta * axis).sum(-1)))
    return slack


def boxes_overlap(
    pose_a: np.ndarray, size_a: np.ndarray, pose_b: np.ndarray, size_b: np.ndarray
) -> np.ndarray:
    """Whether boxes A and B intersect; touching counts."""
    return _separation_slack(pose_a, size_a, pose_b, size_b) >= 0.0


def boxes_penetration(
    pose_a: np.ndarray, size_a: np.ndarray, pose_b: np.ndarray, size_b: np.ndarray
) -> np.ndarray:
    """How deep boxes A and B overlap: the shortest shift that parts them.

    Zero when they do not overlap.
    """
    return np.maximum(_separation_slack(pose_a, size_a, pose_b, size_b), 0.0)


def _points_to_segments(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    """Distance from points ``(..., P, 1, 2)`` to segments ``(..., 1, S, 2)``."""
    edge = end - start
    along = ((points - start) * edge).sum(-1) / np.maximum((edge * edge).sum(-1), 1e-12)
    nearest = start + np.clip(along, 0.0, 1.0)[..., None] * edge
    return np.linalg.norm(points - nearest, axis=-1)


def boxes_gap(
    pose_a: np.ndarray, size_a: np.ndarray, pose_b: np.ndarray, size_b: np.ndarray
) -> np.ndarray:
    """Shortest distance between boxes A and B; zero where they overlap."""
    corners_a = box_corners(pose_a, size_a)
    corners_b = box_corners(pose_b, size_b)
    corners_a, corners_b = np.broadcast_arrays(corners_a, corners_b)

    def corner_to_edges(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        start = polygon[..., None, :, :]
        end = np.roll(polygon, -1, axis=-2)[..., None, :, :]
        return _points_to_segments(points[..., :, None, :], start, end).min(
            axis=(-1, -2)
        )

    gap = np.minimum(
        corner_to_edges(corners_a, corners_b), corner_to_edges(corners_b, corners_a)
    )
    return np.where(boxes_overlap(pose_a, size_a, pose_b, size_b), 0.0, gap)


def to_box_frame(point_xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """A world point in a box's own frame (x forward, y left)."""
    dx, dy = point_xy[0] - pose[0], point_xy[1] - pose[1]
    cos, sin = math.cos(pose[2]), math.sin(pose[2])
    return np.array([dx * cos + dy * sin, -dx * sin + dy * cos])


def impact_zone(point_local: np.ndarray, length: float, width: float) -> dict:
    """Where on a box's outline a point in the box frame lies.

    The point is scaled by the half-extents, so each corner sits at exactly 45
    degrees. ``zone8`` then splits every face into a central part (the middle
    41 % of the face) and the two corner parts it shares with its neighbours;
    ``zone4`` names the face alone. ``u`` and ``v`` are the scaled forward and
    leftward coordinates, kept so the zones can be re-binned.
    """
    u = float(point_local[0]) / (length / 2.0)
    v = float(point_local[1]) / (width / 2.0)
    angle_deg = math.degrees(math.atan2(v, u))
    return {
        "zone8": ZONES8[round(angle_deg / 45.0) % 8],
        "zone4": ZONES4[round(angle_deg / 90.0) % 4],
        "u": round(u, 3),
        "v": round(v, 3),
    }
