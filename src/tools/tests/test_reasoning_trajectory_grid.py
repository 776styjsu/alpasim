# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json
from pathlib import Path

import pytest
from reasoning_coverage.trajectory_grid import (
    trajectory_grid_cell,
    trajectory_grid_items,
    trajectory_grid_items_from_points,
)


def test_grid_rasterizes_between_sparse_waypoints() -> None:
    items = trajectory_grid_items_from_points([(0.0, 0.0, 0.0), (1.0, 5.0, 0.0)])

    assert items == {
        "trajectory_grid:cell=0,0",
        "trajectory_grid:cell=1,0",
        "trajectory_grid:cell=2,0",
        "trajectory_grid:cell=3,0",
    }


def test_grid_is_centered_on_ego_and_clips_at_horizon() -> None:
    centered = trajectory_grid_items_from_points([(0.0, 0.0, -0.9), (1.0, 0.0, 0.9)])
    clipped = trajectory_grid_items_from_points(
        [(0.0, 0.0, 0.0), (5.0, 10.0, 0.0)], horizon_s=3.0
    )

    assert centered == {"trajectory_grid:cell=0,0"}
    assert "trajectory_grid:cell=3,0" in clipped
    assert "trajectory_grid:cell=4,0" not in clipped


def test_grid_loads_rig_frame_waypoints(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    (tmp_path / "additional_info.json").write_text(
        json.dumps(
            {
                "trajectory_source": "trajectory_xy_rig_frame",
                "coordinate_frame": "ego_rig_frame",
                "waypoints": [
                    {
                        "relative_time_s": 0.0,
                        "position_m": {"x": 0.0, "y": 0.0},
                    },
                    {
                        "relative_time_s": 0.1,
                        "position_m": {"x": 0.5, "y": -1.1},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert trajectory_grid_items(metadata_path) == {
        "trajectory_grid:cell=0,-1",
        "trajectory_grid:cell=0,0",
    }


def test_grid_rejects_non_increasing_time() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        trajectory_grid_items_from_points([(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)])


def test_grid_cell_uses_origin_centered_boundaries() -> None:
    assert trajectory_grid_cell(0.9, -0.9) == (0, 0)
    assert trajectory_grid_cell(1.0, -1.0) == (1, 0)
    assert trajectory_grid_cell(-1.1, 1.1) == (-1, 1)
