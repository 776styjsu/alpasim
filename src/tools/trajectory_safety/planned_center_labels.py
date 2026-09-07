# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Label emitted ego plans with center-off-road and recorded-agent outcomes.

The labeler operates on the policy output at each decision step.  It preserves
the emitted 100 ms waypoint sampling, transforms the plan from the ego rig
origin to the AABB center using rollout metadata, and asks whether any future
center within the configured horizon lies outside the union of drivable lane
polygons.  Collision labels are joined from AlpaSim's existing time-aligned
``open_loop_collision`` metric.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import shapely
from alpasim_utils.geometry import pose_from_grpc, trajectory_from_grpc
from alpasim_utils.logs import async_read_pb_log
from omegaconf import OmegaConf
from shapely.geometry.base import BaseGeometry

from .outcomes import load_scene_geometry

LABEL_KIND = "alpamayo_planned_center_safety_labels"
LABEL_HORIZON_S = 3.0
WAYPOINT_INTERVAL_US = 100_000
COLLISION_OUTCOME = "open_loop_collision_with_recorded_agents_3s_100ms"
OFFROAD_OUTCOME = "planned_ego_center_outside_drivable_area_3s_100ms"
COMBINED_OUTCOME = "open_loop_safety_violation_3s_100ms"


@dataclass(frozen=True)
class CenterOffroadResult:
    """Center-containment result for one emitted plan."""

    valid: bool
    invalid_reason: str | None
    initial_center_in_drivable_area: bool
    n_future_waypoints: int
    n_future_waypoints_outside_drivable_area: int
    fraction_future_waypoints_outside_drivable_area: float
    max_center_distance_outside_drivable_area_m: float
    first_center_offroad_time_s: float | None
    planned_center_offroad: bool


def score_planned_centers(
    timestamps_us: np.ndarray,
    positions_xy: np.ndarray,
    *,
    now_us: int,
    horizon_s: float,
    drivable_area: BaseGeometry,
    map_extent: BaseGeometry,
) -> CenterOffroadResult:
    """Score one discrete AABB-center plan without interpolating its waypoints."""
    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    positions = np.asarray(positions_xy, dtype=float)
    if timestamps.ndim != 1:
        raise ValueError("timestamps_us must be one-dimensional")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions_xy must have shape (N, 2)")
    if len(timestamps) != len(positions):
        raise ValueError("timestamps and positions must have equal length")
    if horizon_s <= 0.0:
        raise ValueError("horizon_s must be positive")
    if len(timestamps) == 0:
        return CenterOffroadResult(
            valid=False,
            invalid_reason="empty_plan",
            initial_center_in_drivable_area=False,
            n_future_waypoints=0,
            n_future_waypoints_outside_drivable_area=0,
            fraction_future_waypoints_outside_drivable_area=0.0,
            max_center_distance_outside_drivable_area_m=0.0,
            first_center_offroad_time_s=None,
            planned_center_offroad=False,
        )
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("plan timestamps must be strictly increasing")
    if not np.isfinite(positions).all():
        raise ValueError("plan positions must be finite")
    if drivable_area.is_empty or map_extent.is_empty:
        return CenterOffroadResult(
            valid=False,
            invalid_reason="missing_map_geometry",
            initial_center_in_drivable_area=False,
            n_future_waypoints=0,
            n_future_waypoints_outside_drivable_area=0,
            fraction_future_waypoints_outside_drivable_area=0.0,
            max_center_distance_outside_drivable_area_m=0.0,
            first_center_offroad_time_s=None,
            planned_center_offroad=False,
        )

    current_indices = np.flatnonzero(timestamps == now_us)
    if len(current_indices) != 1:
        return CenterOffroadResult(
            valid=False,
            invalid_reason="missing_unique_current_waypoint",
            initial_center_in_drivable_area=False,
            n_future_waypoints=0,
            n_future_waypoints_outside_drivable_area=0,
            fraction_future_waypoints_outside_drivable_area=0.0,
            max_center_distance_outside_drivable_area_m=0.0,
            first_center_offroad_time_s=None,
            planned_center_offroad=False,
        )

    current_point = shapely.Point(positions[current_indices[0]])
    initial_inside = bool(shapely.covers(drivable_area, current_point))
    horizon_us = round(horizon_s * 1e6)
    future_mask = (timestamps > now_us) & (timestamps <= now_us + horizon_us)
    future_timestamps = timestamps[future_mask]
    future_positions = positions[future_mask]
    if len(future_timestamps) == 0:
        return CenterOffroadResult(
            valid=False,
            invalid_reason="no_future_waypoint_in_horizon",
            initial_center_in_drivable_area=initial_inside,
            n_future_waypoints=0,
            n_future_waypoints_outside_drivable_area=0,
            fraction_future_waypoints_outside_drivable_area=0.0,
            max_center_distance_outside_drivable_area_m=0.0,
            first_center_offroad_time_s=None,
            planned_center_offroad=False,
        )

    future_points = shapely.points(future_positions[:, 0], future_positions[:, 1])
    if not bool(np.asarray(shapely.covers(map_extent, future_points)).all()):
        return CenterOffroadResult(
            valid=False,
            invalid_reason="future_waypoint_outside_map_extent",
            initial_center_in_drivable_area=initial_inside,
            n_future_waypoints=len(future_timestamps),
            n_future_waypoints_outside_drivable_area=0,
            fraction_future_waypoints_outside_drivable_area=0.0,
            max_center_distance_outside_drivable_area_m=0.0,
            first_center_offroad_time_s=None,
            planned_center_offroad=False,
        )
    if not initial_inside:
        return CenterOffroadResult(
            valid=False,
            invalid_reason="initial_center_outside_drivable_area",
            initial_center_in_drivable_area=False,
            n_future_waypoints=len(future_timestamps),
            n_future_waypoints_outside_drivable_area=0,
            fraction_future_waypoints_outside_drivable_area=0.0,
            max_center_distance_outside_drivable_area_m=0.0,
            first_center_offroad_time_s=None,
            planned_center_offroad=False,
        )

    inside = np.asarray(shapely.covers(drivable_area, future_points), dtype=bool)
    outside = ~inside
    distances = np.asarray(shapely.distance(drivable_area, future_points), dtype=float)
    outside_count = int(outside.sum())
    first_offroad_time_s = (
        float((future_timestamps[np.flatnonzero(outside)[0]] - now_us) / 1e6)
        if outside_count
        else None
    )
    return CenterOffroadResult(
        valid=True,
        invalid_reason=None,
        initial_center_in_drivable_area=True,
        n_future_waypoints=len(future_timestamps),
        n_future_waypoints_outside_drivable_area=outside_count,
        fraction_future_waypoints_outside_drivable_area=float(outside.mean()),
        max_center_distance_outside_drivable_area_m=float(distances.max()),
        first_center_offroad_time_s=first_offroad_time_s,
        planned_center_offroad=bool(outside_count),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _map_extent(geometry_path: Path) -> BaseGeometry:
    with geometry_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    coordinates = [
        coordinate[:2]
        for line in payload["map_linestrings"]
        for coordinate in line["coords"]
    ]
    if not coordinates:
        return shapely.Polygon()
    array = np.asarray(coordinates, dtype=float)
    return shapely.box(
        float(array[:, 0].min()),
        float(array[:, 1].min()),
        float(array[:, 0].max()),
        float(array[:, 1].max()),
    )


def _frame_index(scene_path: Path) -> tuple[dict[int, str], list[Path]]:
    metadata_paths = sorted(scene_path.glob("extracted_frames/**/metadata.json"))
    if not metadata_paths:
        raise ValueError(f"No extracted metadata under {scene_path}")
    by_time: dict[int, str] = {}
    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        timestamp = metadata["time_now_us"]
        frame_id = metadata["clip_id"]
        if not isinstance(timestamp, int) or not isinstance(frame_id, str):
            raise ValueError(f"Invalid frame identity in {metadata_path}")
        if timestamp in by_time:
            raise ValueError(f"Duplicate frame timestamp {timestamp} in {scene_path}")
        by_time[timestamp] = frame_id
    return by_time, metadata_paths


def _metric_index(metrics_path: Path) -> dict[int, tuple[bool, bool]]:
    metric = pl.read_parquet(metrics_path).filter(
        pl.col("name") == "open_loop_collision"
    )
    if metric.height == 0:
        raise ValueError(f"No open_loop_collision rows in {metrics_path}")
    result: dict[int, tuple[bool, bool]] = {}
    for timestamp, value, valid in metric.select(
        "timestamps_us", "values", "valid"
    ).iter_rows():
        timestamp_int = int(timestamp)
        if timestamp_int in result:
            raise ValueError(f"Duplicate open-loop collision timestamp {timestamp_int}")
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite collision value at {timestamp_int}")
        result[timestamp_int] = (bool(float(value) > 0.0), bool(valid))
    return result


async def _label_asl(
    asl_path: Path,
    frame_by_time: dict[int, str],
    collision_by_time: dict[int, tuple[bool, bool]],
    drivable_area: BaseGeometry,
    map_extent: BaseGeometry,
    horizon_s: float,
) -> list[dict[str, Any]]:
    rig_to_center = None
    pending_now_us: int | None = None
    rows: list[dict[str, Any]] = []
    async for message in async_read_pb_log(str(asl_path)):
        message_type = message.WhichOneof("log_entry")
        if message_type == "rollout_metadata":
            rig_to_center = pose_from_grpc(
                message.rollout_metadata.transform_ego_coords_rig_to_aabb
            )
        elif message_type == "driver_request":
            pending_now_us = int(message.driver_request.time_now_us)
        elif message_type == "driver_return":
            if pending_now_us is None:
                continue
            now_us = pending_now_us
            pending_now_us = None
            if now_us not in frame_by_time:
                continue
            if rig_to_center is None:
                raise ValueError(
                    f"Driver response precedes rollout metadata: {asl_path}"
                )
            trajectory = trajectory_from_grpc(
                message.driver_return.trajectory
            ).transform(rig_to_center, is_relative=True)
            if not np.all(np.diff(trajectory.timestamps_us) == WAYPOINT_INTERVAL_US):
                raise ValueError(
                    f"Plan at {now_us} in {asl_path} is not sampled every 100 ms"
                )
            offroad = score_planned_centers(
                trajectory.timestamps_us,
                trajectory.positions[:, :2],
                now_us=now_us,
                horizon_s=horizon_s,
                drivable_area=drivable_area,
                map_extent=map_extent,
            )
            try:
                collision, collision_valid = collision_by_time[now_us]
            except KeyError as exc:
                raise ValueError(
                    f"Missing collision label at {now_us} in {asl_path}"
                ) from exc
            valid = offroad.valid and collision_valid
            rows.append(
                {
                    "frame_id": frame_by_time[now_us],
                    "time_now_us": now_us,
                    **asdict(offroad),
                    "open_loop_collision_with_recorded_agents": collision,
                    "open_loop_collision_valid": collision_valid,
                    "open_loop_safety_violation": (
                        (collision or offroad.planned_center_offroad)
                        if valid
                        else False
                    ),
                    "valid": valid,
                }
            )
    return rows


def _label_scene(arguments: tuple[str, float]) -> dict[str, Any]:
    scene_path = Path(arguments[0])
    horizon_s = arguments[1]
    metrics_paths = list(scene_path.glob("rollouts/*/*/metrics.parquet"))
    if len(metrics_paths) != 1:
        raise ValueError(
            f"Expected one metrics.parquet under {scene_path}, found {len(metrics_paths)}"
        )
    metrics_path = metrics_paths[0]
    asl_path = metrics_path.with_name("rollout.asl")
    if not asl_path.exists():
        raise ValueError(f"Missing ASL sibling: {asl_path}")
    eval_config_path = scene_path / "eval-config.yaml"
    eval_config = OmegaConf.load(eval_config_path)
    collision_horizon_s = float(eval_config.scorers.open_loop_collision.horizon_s)
    if not math.isclose(collision_horizon_s, horizon_s):
        raise ValueError(
            f"Collision metric horizon {collision_horizon_s} in {eval_config_path} "
            f"does not match requested horizon {horizon_s}"
        )
    frame_by_time, metadata_paths = _frame_index(scene_path)
    geometry_paths = [
        metadata_path.with_name("trajectory_plot_geometry.json")
        for metadata_path in metadata_paths
    ]
    if not all(path.exists() for path in geometry_paths):
        raise ValueError(f"Missing trajectory plot geometry under {scene_path}")
    geometry_path = geometry_paths[0]
    geometry = load_scene_geometry(geometry_path)
    extent = _map_extent(geometry_path)
    collision_by_time = _metric_index(metrics_path)
    rows = asyncio.run(
        _label_asl(
            asl_path,
            frame_by_time,
            collision_by_time,
            geometry.drivable,
            extent,
            horizon_s,
        )
    )
    if len(rows) != len(frame_by_time):
        labeled = {row["frame_id"] for row in rows}
        missing = sorted(set(frame_by_time.values()) - labeled)
        raise ValueError(
            f"Labeled {len(rows)}/{len(frame_by_time)} frames in {scene_path}; "
            f"missing {missing[:5]}"
        )
    scene_id = rows[0]["frame_id"].rsplit("_t", maxsplit=1)[0]
    if any(row["frame_id"].rsplit("_t", maxsplit=1)[0] != scene_id for row in rows):
        raise ValueError(f"Multiple scene IDs under {scene_path}")
    return {
        "scene_id": scene_id,
        "scene_path": str(scene_path),
        "asl_path": str(asl_path),
        "asl_size_bytes": asl_path.stat().st_size,
        "asl_mtime_ns": asl_path.stat().st_mtime_ns,
        "metrics_path": str(metrics_path),
        "metrics_sha256": _file_sha256(metrics_path),
        "eval_config_path": str(eval_config_path),
        "eval_config_sha256": _file_sha256(eval_config_path),
        "map_geometry_path": str(geometry_path),
        "map_geometry_sha256": _file_sha256(geometry_path),
        "n_drivable_lanes": geometry.n_lanes,
        "frames": rows,
    }


def build_labels(root: Path, *, horizon_s: float, workers: int) -> dict[str, Any]:
    """Build exhaustive labels for every extracted decision under ``root``."""
    if not math.isclose(horizon_s, LABEL_HORIZON_S):
        raise ValueError(
            f"horizon_s must be {LABEL_HORIZON_S} to match the stored collision metric"
        )
    if workers <= 0:
        raise ValueError("workers must be positive")
    scene_paths = sorted(
        path
        for path in root.glob("scene_*")
        if next(path.glob("extracted_frames/**/metadata.json"), None) is not None
    )
    if not scene_paths:
        raise ValueError(f"No extracted scenes under {root}")
    arguments = [(str(path), horizon_s) for path in scene_paths]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        scenes = list(executor.map(_label_scene, arguments))
    frames = [row for scene in scenes for row in scene.pop("frames")]
    frame_ids = [row["frame_id"] for row in frames]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Duplicate frame IDs across labeled scenes")
    invalid_reasons = Counter(
        row["invalid_reason"] for row in frames if not row["valid"]
    )
    valid = [row for row in frames if row["valid"]]
    return {
        "schema_version": 1,
        "kind": LABEL_KIND,
        "inputs": {
            "root": str(root),
            "labeler_sha256": _file_sha256(Path(__file__)),
            "scenes": scenes,
        },
        "config": {
            "horizon_s": horizon_s,
            "sampling": "emitted_waypoints_only",
            "expected_waypoint_interval_s": 0.1,
            "ego_reference_point": "aabb_center_from_rollout_rig_transform",
            "drivable_definition": "union_of_road_lane_polygons",
            "initial_center_must_be_drivable": True,
            "future_waypoints_must_be_within_map_extent": True,
            "collision_source_metric": "open_loop_collision",
            "outcome_names": {
                "collision": COLLISION_OUTCOME,
                "offroad": OFFROAD_OUTCOME,
                "combined": COMBINED_OUTCOME,
            },
        },
        "dataset_audit": {
            "n_scenes": len(scenes),
            "n_frames": len(frames),
            "n_valid_frames": len(valid),
            "n_invalid_frames": len(frames) - len(valid),
            "invalid_reasons": dict(sorted(invalid_reasons.items())),
            "n_collision_positive_valid": sum(
                row["open_loop_collision_with_recorded_agents"] for row in valid
            ),
            "n_offroad_positive_valid": sum(
                row["planned_center_offroad"] for row in valid
            ),
            "n_combined_positive_valid": sum(
                row["open_loop_safety_violation"] for row in valid
            ),
        },
        "frames": sorted(frames, key=lambda row: row["frame_id"]),
        "limitations": [
            "Outcomes describe the emitted discrete plan, not executed closed-loop motion.",
            "Collision uses recorded, non-reactive agent futures.",
            "No interpolation is added between the emitted 100 ms waypoints.",
            "Center containment is a high-precision proxy and misses partial-footprint departures.",
            "Lane polygons and their rectangular map extent come from extracted visualization geometry.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label discrete Alpamayo plans with prospective safety proxies"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/project/AV/drive-the-thoughts/alpasim/"
            "tutorial_alpamayo_ood_hazard_2604"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/reasoning_coverage/" "ood_hazard_2604.planned_safety_3s_100ms.json"
        ),
    )
    parser.add_argument("--horizon-s", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_labels(args.root, horizon_s=args.horizon_s, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result["dataset_audit"], indent=2))
    print(f"Labels: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
