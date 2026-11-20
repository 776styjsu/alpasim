# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for StepEvent lifecycle management."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest
from alpasim_runtime.events.base import EventQueue
from alpasim_runtime.events.state import RolloutState, ServiceBundle, StepContext
from alpasim_runtime.events.step import InitialStepEvent, StepEvent
from alpasim_utils.geometry import DynamicTrajectory, Pose, Trajectory


def _make_pose(x: float, y: float, z: float) -> Pose:
    return Pose(
        np.array([x, y, z], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


def _make_trajectory(start_us: int, stop_us: int, step_us: int) -> Trajectory:
    timestamps = list(range(start_us, stop_us + 1, step_us))
    poses = [_make_pose(float(t) / 1e6, 0.0, 0.0) for t in timestamps]
    return Trajectory.from_poses(
        timestamps=np.array(timestamps, dtype=np.uint64),
        poses=poses,
    )


class TestInitialStepEvent:
    """InitialStepEvent logs the rollout's starting state."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("first_policy_timestamp_us", "expected_timestamps_us"),
        [
            (200_000, [0, 200_000]),
            # No cameras: the policy starts at the context start.
            (0, [0]),
        ],
    )
    async def test_logs_each_starting_timestamp_once(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_broadcaster: AsyncMock,
        first_policy_timestamp_us: int,
        expected_timestamps_us: list[int],
    ) -> None:
        rollout_state.unbound.egomotion_context_start_us = 0
        rollout_state.unbound.first_policy_timestamp_us = first_policy_timestamp_us
        event = InitialStepEvent(timestamp_us=0, services=service_bundle)

        await event.handle(rollout_state, EventQueue())

        logged_timestamps_us = [
            call.args[0].actor_poses.timestamp_us
            for call in mock_broadcaster.broadcast.call_args_list
        ]
        assert logged_timestamps_us == expected_timestamps_us


class TestStepEventInitialCase:
    """StepEvent at scene_start_us with no prior StepContext (bootstrap)."""

    @pytest.mark.asyncio
    async def test_creates_step_context_when_none(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
    ) -> None:
        rollout_state.step_context = None
        event = StepEvent(
            timestamp_us=0,
            control_timestep_us=100_000,
            services=service_bundle,
        )
        queue = EventQueue()

        await event.run(rollout_state, queue)

        assert rollout_state.step_context is not None
        assert rollout_state.step_context.outstanding_tasks == []

    @pytest.mark.asyncio
    async def test_logs_initial_actor_poses(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_broadcaster: AsyncMock,
    ) -> None:
        rollout_state.step_context = None
        event = StepEvent(
            timestamp_us=0,
            control_timestep_us=100_000,
            services=service_bundle,
        )
        queue = EventQueue()

        await event.run(rollout_state, queue)

        mock_broadcaster.broadcast.assert_called()

    @pytest.mark.asyncio
    async def test_seeds_step_wall_start(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
    ) -> None:
        rollout_state.step_context = None
        rollout_state.step_wall_start = 0.0
        event = StepEvent(
            timestamp_us=0,
            control_timestep_us=100_000,
            services=service_bundle,
        )
        queue = EventQueue()

        await event.run(rollout_state, queue)

        assert rollout_state.step_wall_start > 0.0


class TestStepEventNormalCase:
    """StepEvent during steady-state simulation with a filled StepContext."""

    @pytest.mark.asyncio
    async def test_replaces_step_context_with_fresh_one(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
    ) -> None:
        """After committing, step_context is a fresh StepContext (not None)."""
        # State ego_trajectory ends at 200_000; corrected must start after that.
        ego_traj = _make_trajectory(300_000, 300_000, 100_000)
        zero_dynamics = np.zeros((len(ego_traj), 12), dtype=np.float64)
        dynamic_traj = DynamicTrajectory.from_trajectory_and_dynamics(
            ego_traj, zero_dynamics
        )

        old_ctx = StepContext(step_start_us=200_000, target_time_us=300_000)
        old_ctx.ego_true = dynamic_traj
        old_ctx.ego_estimated = dynamic_traj
        old_ctx.corrected_ego_trajectory = ego_traj
        rollout_state.step_context = old_ctx

        event = StepEvent(
            timestamp_us=200_000,
            control_timestep_us=100_000,
            services=service_bundle,
        )
        queue = EventQueue()
        await event.run(rollout_state, queue)

        assert rollout_state.step_context is not old_ctx
        assert rollout_state.step_context is not None
        assert rollout_state.step_context.outstanding_tasks == []
        assert rollout_state.step_context.step_start_us == 0  # fresh defaults
