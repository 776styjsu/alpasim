# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Trajectory feature computation for CoT consistency analysis.

Computes raw numerical trajectory features (positions, velocities,
accelerations, lateral deviation) from rig-frame trajectory poses.
The LLM interprets maneuvers from the raw data — no categorical
labels or magic thresholds.
"""

import numpy as np


def compute_trajectory_features(
    trajectory_xy: np.ndarray,
    dt: float = 0.1,
) -> dict:
    """Compute raw numerical trajectory features for LLM evaluation.

    Args:
        trajectory_xy: Trajectory positions in rig frame, shape (T, 2) or (T, 3).
            X = forward, Y = left.
        dt: Time between steps in seconds (default 0.1 = 10Hz).

    Returns:
        Dict with:
            - summary_stats: Aggregate trajectory statistics
            - table_rows: Per-timestep data (downsampled)
            - markdown_kv: Formatted string for the LLM prompt
    """
    if trajectory_xy is None or len(trajectory_xy) < 3:
        return {"error": "Trajectory too short"}

    xy = np.asarray(trajectory_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return {"error": f"Invalid trajectory shape: {xy.shape}"}

    xy = xy[:, :2]  # only x, y
    T = xy.shape[0]

    # --- Per-step kinematics ---
    dxy = np.diff(xy, axis=0)  # (T-1, 2)
    velocities = dxy / dt  # m/s
    speeds = np.linalg.norm(velocities, axis=1)  # (T-1,)

    # Acceleration (from speed differences)
    if len(speeds) > 1:
        accel = np.diff(speeds) / dt  # m/s², (T-2,)
    else:
        accel = np.array([0.0])

    # Lateral velocity (Y component in rig/ego frame)
    lat_vel = velocities[:, 1]  # positive = leftward

    # --- Summary stats ---
    total_path_length = float(np.sum(np.linalg.norm(dxy, axis=1)))
    final_displacement = xy[-1] - xy[0]
    max_lat_deviation = float(xy[:, 1].max())
    min_lat_deviation = float(xy[:, 1].min())

    summary_stats = {
        "duration_s": round((T - 1) * dt, 1),
        "total_path_length_m": round(total_path_length, 2),
        "final_longitudinal_m": round(float(final_displacement[0]), 2),
        "final_lateral_m": round(float(final_displacement[1]), 2),
        "max_lateral_m": round(max_lat_deviation, 2),
        "min_lateral_m": round(min_lat_deviation, 2),
        "mean_speed_ms": round(float(speeds.mean()), 2),
        "max_speed_ms": round(float(speeds.max()), 2),
        "min_speed_ms": round(float(speeds.min()), 2),
        "mean_accel_ms2": round(float(accel.mean()), 2),
    }

    # --- Downsample to ~1.25Hz for the LLM table ---
    step = max(1, int(0.8 / dt))  # every 8 steps at 10Hz
    sample_indices = list(range(0, T, step))
    if (T - 1) not in sample_indices:
        sample_indices.append(T - 1)

    table_rows = []
    for i in sample_indices:
        t_sec = round(i * dt, 1)
        x, y = round(float(xy[i, 0]), 2), round(float(xy[i, 1]), 2)
        if i < len(speeds):
            spd = round(float(speeds[i]), 2)
            lv = round(float(lat_vel[i]), 2)
        else:
            spd = round(float(speeds[-1]), 2)
            lv = round(float(lat_vel[-1]), 2)
        if i < len(accel):
            acc = round(float(accel[i]), 2)
        elif len(accel) > 0:
            acc = round(float(accel[-1]), 2)
        else:
            acc = 0.0
        table_rows.append(
            {"t": t_sec, "x": x, "y": y, "speed": spd, "lat_vel": lv, "accel": acc}
        )

    # Format as markdown for the prompt
    lines = []
    for r in table_rows:
        lines.append(f"### t={r['t']}s")
        lines.append(f"- x: {r['x']}m")
        lines.append(f"- y: {r['y']}m")
        lines.append(f"- speed: {r['speed']} m/s")
        lines.append(f"- lat_vel: {r['lat_vel']} m/s")
        lines.append(f"- accel: {r['accel']} m/s²")
    markdown_kv = "\n".join(lines)

    return {
        "summary_stats": summary_stats,
        "table_rows": table_rows,
        "markdown_kv": markdown_kv,
    }
