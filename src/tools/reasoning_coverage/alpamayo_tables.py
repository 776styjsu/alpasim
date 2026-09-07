# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Closed portions of the published Alpamayo-R1 reasoning vocabulary."""

from __future__ import annotations

TABLE1_LONGITUDINAL_DECISIONS = (
    "none",
    "set_speed_tracking",
    "lead_obstacle_following",
    "speed_adaptation_road_events",
    "gap_searching_for_lane_change_merge_zipper",
    "acceleration_for_passing_overtaking",
    "yield_agent_right_of_way",
    "stop_for_static_constraints",
)

TABLE1_LATERAL_DECISIONS = (
    "none",
    "lane_keeping_centering",
    "merge_split_facility_change",
    "out_of_lane_nudge",
    "in_lane_nudge",
    "lane_change",
    "pull_over_curb_approach",
    "turn",
    "lateral_maneuver_abort",
)

TABLE2_COMPONENT_CATEGORIES = (
    "critical_objects",
    "traffic_lights",
    "stop_yield_control",
    "road_events",
    "lane_lanelines",
    "routing_intent",
    "odd_constraints",
)

TABLE5_LONGITUDINAL_ACTIONS = (
    "gentle_accelerate",
    "strong_accelerate",
    "gentle_decelerate",
    "strong_decelerate",
    "maintain_speed",
    "stop",
    "reverse",
)

TABLE5_LATERAL_ACTIONS = (
    "steer_left",
    "steer_right",
    "sharp_steer_left",
    "sharp_steer_right",
    "reverse_left",
    "reverse_right",
    "go_straight",
)


def table1_decision_pairs() -> tuple[str, ...]:
    """Return all 71 non-empty longitudinal/lateral Table 1 pairs."""
    return tuple(
        f"{longitudinal}|{lateral}"
        for longitudinal in TABLE1_LONGITUDINAL_DECISIONS
        for lateral in TABLE1_LATERAL_DECISIONS
        if (longitudinal, lateral) != ("none", "none")
    )


def table5_action_pairs() -> tuple[str, ...]:
    """Return the 49 simultaneous longitudinal/lateral atomic-action pairs."""
    return tuple(
        f"{longitudinal}|{lateral}"
        for longitudinal in TABLE5_LONGITUDINAL_ACTIONS
        for lateral in TABLE5_LATERAL_ACTIONS
    )


def table2_category_x_table1_decision_pairs() -> tuple[str, ...]:
    """Return the 497 Table 2 category by non-empty Table 1 pair products."""
    return tuple(
        f"{category}|{decision_pair}"
        for category in TABLE2_COMPONENT_CATEGORIES
        for decision_pair in table1_decision_pairs()
    )
