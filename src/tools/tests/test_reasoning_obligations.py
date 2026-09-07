# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import math

from reasoning_coverage.obligations import (
    REASONING_STATUSES,
    ActorState,
    alpamayo_decision_pairs,
    applicable_obligations,
    catalog,
    denominator_items,
    program_decision_pairs,
    reasoning_status,
)
from reasoning_coverage.program import compile_reasoning


def _actor(
    actor_id: str,
    actor_label: str,
    x_m: float,
    y_m: float,
    yaw_rad: float,
    velocity_x_mps: float,
    velocity_y_mps: float,
) -> ActorState:
    return ActorState(
        actor_id=actor_id,
        actor_label=actor_label,
        x_m=x_m,
        y_m=y_m,
        yaw_rad=yaw_rad,
        velocity_x_mps=velocity_x_mps,
        velocity_y_mps=velocity_y_mps,
        length_m=4.5,
        width_m=2.0,
    )


def test_catalog_is_a_closed_feasible_denominator() -> None:
    definitions = catalog()

    assert len(definitions) == 27
    assert len({item.obligation_id for item in definitions}) == len(definitions)
    assert all(item.distance_band in {"near", "mid", "far"} for item in definitions)
    assert len(REASONING_STATUSES) == 4
    assert len(alpamayo_decision_pairs()) == 71
    assert len(denominator_items(include_status=True)) > len(
        denominator_items(include_status=False)
    )
    decision_denominator = denominator_items(
        include_status=True, include_decisions=True
    )
    assert (
        sum(item.startswith("cro-decision:variant=") for item in decision_denominator)
        == 27 * 71
    )


def test_applicability_selects_nearest_aligned_lead() -> None:
    ego = _actor("EGO", "", 0.0, 0.0, 0.0, 10.0, 0.0)
    near = _actor("near", "automobile", 10.0, 0.2, 0.0, 5.0, 0.0)
    far = _actor("far", "heavy_truck", 30.0, 0.0, 0.0, 0.0, 0.0)

    obligations = applicable_obligations((ego, far, near))

    assert len(obligations) == 1
    obligation = obligations[0]
    assert obligation.actor_id == "near"
    assert obligation.obligation_id == (
        "lead_agent_following:light_vehicle:slower:near"
    )


def test_applicability_detects_constant_velocity_crossing_conflict() -> None:
    ego = _actor("EGO", "", 0.0, 0.0, 0.0, 5.0, 0.0)
    pedestrian = _actor("person", "person", 10.0, -10.0, math.pi / 2.0, 0.0, 5.0)

    obligations = applicable_obligations((ego, pedestrian))

    assert len(obligations) == 1
    obligation = obligations[0]
    assert obligation.obligation_id == "crossing_agent_yield:vru:moving:near"
    assert obligation.time_to_closest_approach_s == 2.0
    assert obligation.distance_at_closest_approach_m == 0.0


def test_reasoning_status_requires_grounding_and_expected_decision() -> None:
    ego = _actor("EGO", "", 0.0, 0.0, 0.0, 10.0, 0.0)
    lead = _actor("lead", "automobile", 10.0, 0.0, 0.0, 5.0, 0.0)
    obligation = applicable_obligations((ego, lead))[0]

    grounded = compile_reasoning(
        "Keep distance to the lead vehicle since it is directly ahead in our lane"
    )
    ungrounded = compile_reasoning("Maintain a safe distance")
    unrelated = compile_reasoning("Keep centered in the lane")

    assert reasoning_status(grounded, obligation) == "recognized_expected_decision"
    assert reasoning_status(ungrounded, obligation) == (
        "expected_decision_without_recognition"
    )
    assert reasoning_status(unrelated, obligation) == "exercised_not_recognized"
    assert program_decision_pairs(grounded) == frozenset(
        {"lead_obstacle_following|none"}
    )
