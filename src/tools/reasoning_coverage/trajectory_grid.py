# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Outcome-blind ego-centric grid coverage for emitted trajectories."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CELL_SIZE_M = 2.0
DEFAULT_HORIZON_S = 3.0
DEFAULT_RASTER_STEP_M = 0.5


def _finite_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Trajectory waypoint {name} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"Trajectory waypoint {name} is not finite: {value!r}")
    return number


def _cell_index(value: float, cell_size_m: float) -> int:
    """Index an origin-centered grid with boundaries at half-cell offsets."""
    return math.floor(value / cell_size_m + 0.5)


def trajectory_grid_cell(
    x_m: float,
    y_m: float,
    *,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
) -> tuple[int, int]:
    """Return the origin-centered cell containing an ego-frame position."""
    if cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")
    if not math.isfinite(x_m) or not math.isfinite(y_m):
        raise ValueError("Grid position must be finite")
    return _cell_index(x_m, cell_size_m), _cell_index(y_m, cell_size_m)


def trajectory_grid_items_from_points(
    points: Iterable[tuple[float, float, float]],
    *,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    horizon_s: float = DEFAULT_HORIZON_S,
    raster_step_m: float = DEFAULT_RASTER_STEP_M,
) -> set[str]:
    """Rasterize ``(time_s, x_m, y_m)`` points into ego-centric grid cells."""
    if cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")
    if horizon_s < 0.0:
        raise ValueError("horizon_s must be non-negative")
    if raster_step_m <= 0.0 or raster_step_m > cell_size_m:
        raise ValueError("raster_step_m must be positive and at most cell_size_m")

    materialized = list(points)
    for index, point in enumerate(materialized):
        if len(point) != 3:
            raise ValueError(f"Trajectory point {index} must contain time, x, and y")
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"Trajectory point {index} contains a non-finite value")
        if point[0] < 0.0:
            raise ValueError(f"Trajectory point {index} has negative relative time")
        if index and point[0] <= materialized[index - 1][0]:
            raise ValueError("Trajectory point times must be strictly increasing")
    if not materialized or materialized[0][0] > horizon_s:
        return set()

    clipped: list[tuple[float, float, float]] = []
    for index, point in enumerate(materialized):
        time_s, x_m, y_m = point
        if time_s <= horizon_s:
            clipped.append(point)
            continue
        previous_time, previous_x, previous_y = materialized[index - 1]
        if clipped and previous_time < horizon_s:
            fraction = (horizon_s - previous_time) / (time_s - previous_time)
            clipped.append(
                (
                    horizon_s,
                    previous_x + fraction * (x_m - previous_x),
                    previous_y + fraction * (y_m - previous_y),
                )
            )
        break

    cells: set[tuple[int, int]] = set()

    def add_point(x_m: float, y_m: float) -> None:
        cells.add(trajectory_grid_cell(x_m, y_m, cell_size_m=cell_size_m))

    add_point(clipped[0][1], clipped[0][2])
    for (_, start_x, start_y), (_, end_x, end_y) in zip(
        clipped, clipped[1:], strict=False
    ):
        distance = math.hypot(end_x - start_x, end_y - start_y)
        steps = max(1, math.ceil(distance / raster_step_m))
        for step in range(1, steps + 1):
            fraction = step / steps
            add_point(
                start_x + fraction * (end_x - start_x),
                start_y + fraction * (end_y - start_y),
            )
    return {f"trajectory_grid:cell={x},{y}" for x, y in cells}


def trajectory_grid_items(
    metadata_path: Path | None,
    *,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    horizon_s: float = DEFAULT_HORIZON_S,
    raster_step_m: float = DEFAULT_RASTER_STEP_M,
) -> set[str]:
    """Load an emitted rig-frame trajectory and return its occupied grid cells."""
    if metadata_path is None:
        return set()
    info_path = metadata_path.with_name("additional_info.json")
    try:
        with info_path.open("r", encoding="utf-8") as file:
            info = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return set()

    if info.get("coordinate_frame") != "ego_rig_frame":
        raise ValueError(f"Expected ego_rig_frame trajectory in {info_path}")
    if info.get("trajectory_source") != "trajectory_xy_rig_frame":
        raise ValueError(f"Expected trajectory_xy_rig_frame in {info_path}")
    raw_waypoints = info.get("waypoints")
    if not isinstance(raw_waypoints, list):
        raise ValueError(f"Expected waypoint list in {info_path}")

    points: list[tuple[float, float, float]] = []
    for index, waypoint in enumerate(raw_waypoints):
        if not isinstance(waypoint, dict) or not isinstance(
            waypoint.get("position_m"), dict
        ):
            raise ValueError(f"Malformed trajectory waypoint {index} in {info_path}")
        position = waypoint["position_m"]
        points.append(
            (
                _finite_float(waypoint.get("relative_time_s"), name="time"),
                _finite_float(position.get("x"), name="x"),
                _finite_float(position.get("y"), name="y"),
            )
        )
    return trajectory_grid_items_from_points(
        points,
        cell_size_m=cell_size_m,
        horizon_s=horizon_s,
        raster_step_m=raster_step_m,
    )
