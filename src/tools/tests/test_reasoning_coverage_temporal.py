# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import pytest
from reasoning_coverage.temporal_sensitivity import (
    _tied_random_orders,
    common_strict_pre_event_frames,
    prefix_frame_count,
)


def test_prefix_frame_count_rounds_up() -> None:
    assert prefix_frame_count(182, 0.25) == 46
    with pytest.raises(ValueError, match="fraction"):
        prefix_frame_count(182, 0.0)


def test_common_pre_event_horizon_is_equal_and_strict() -> None:
    frame_times = {
        "early": (10, 20, 30, 40),
        "late": (10, 20, 30, 40),
        "safe": (10, 20, 30, 40),
    }
    events = {"early": 10, "late": 35}

    assert common_strict_pre_event_frames(frame_times, events) == 0


def test_zero_frame_orders_are_identical_across_arms() -> None:
    orders = _tied_random_orders(["c", "a", "b"], repeats=3, seed=7)

    assert all(orders[arm] == orders["random"] for arm in orders)
    assert len({tuple(order) for order in orders["random"]}) > 1
