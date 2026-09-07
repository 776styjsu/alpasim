# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
import shapely
from trajectory_safety.planned_center_labels import score_planned_centers


def _timestamps() -> np.ndarray:
    return np.arange(0, 4_100_000, 100_000, dtype=np.int64)


def test_center_offroad_uses_only_emitted_future_waypoints_in_horizon() -> None:
    timestamps = _timestamps()
    positions = np.column_stack(
        (
            np.linspace(0.0, 40.0, len(timestamps)),
            np.where(timestamps >= 2_000_000, 3.0, 0.0),
        )
    )
    result = score_planned_centers(
        timestamps,
        positions,
        now_us=0,
        horizon_s=3.0,
        drivable_area=shapely.box(-5.0, -2.0, 50.0, 2.0),
        map_extent=shapely.box(-10.0, -10.0, 60.0, 10.0),
    )

    assert result.valid
    assert result.n_future_waypoints == 30
    assert result.n_future_waypoints_outside_drivable_area == 11
    assert result.first_center_offroad_time_s == 2.0
    assert result.max_center_distance_outside_drivable_area_m == 1.0
    assert result.planned_center_offroad


def test_center_on_drivable_boundary_is_not_offroad() -> None:
    timestamps = _timestamps()
    positions = np.column_stack(
        (np.linspace(0.0, 40.0, len(timestamps)), np.full(len(timestamps), 2.0))
    )
    result = score_planned_centers(
        timestamps,
        positions,
        now_us=0,
        horizon_s=3.0,
        drivable_area=shapely.box(-5.0, -2.0, 50.0, 2.0),
        map_extent=shapely.box(-10.0, -10.0, 60.0, 10.0),
    )

    assert result.valid
    assert not result.planned_center_offroad


def test_plan_starting_offroad_is_invalid_not_positive() -> None:
    timestamps = _timestamps()
    positions = np.column_stack(
        (np.linspace(0.0, 40.0, len(timestamps)), np.full(len(timestamps), 3.0))
    )
    result = score_planned_centers(
        timestamps,
        positions,
        now_us=0,
        horizon_s=3.0,
        drivable_area=shapely.box(-5.0, -2.0, 50.0, 2.0),
        map_extent=shapely.box(-10.0, -10.0, 60.0, 10.0),
    )

    assert not result.valid
    assert result.invalid_reason == "initial_center_outside_drivable_area"
    assert not result.planned_center_offroad


def test_waypoint_beyond_available_map_extent_invalidates_plan() -> None:
    timestamps = _timestamps()
    positions = np.column_stack(
        (np.linspace(0.0, 40.0, len(timestamps)), np.zeros(len(timestamps)))
    )
    result = score_planned_centers(
        timestamps,
        positions,
        now_us=0,
        horizon_s=3.0,
        drivable_area=shapely.box(-5.0, -2.0, 50.0, 2.0),
        map_extent=shapely.box(-10.0, -10.0, 20.0, 10.0),
    )

    assert not result.valid
    assert result.invalid_reason == "future_waypoint_outside_map_extent"
