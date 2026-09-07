# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from reasoning_coverage.experiment_analysis import (
    _concentration,
    _coverage_curve_summary,
    _curve_summary,
    _descending_score_orders,
    _detection_ranks,
    _exclusive_summary,
    _fault_unit_apfd,
    _inconsistency_mode,
    _mean_fault_value_by_cell,
    _trajectory_grid_cell,
    _unsafe_mode,
)


def test_detection_rank_is_the_position_of_each_positive_frame() -> None:
    units = {"a1": frozenset(("a1",)), "b1": frozenset(("b1",))}

    ranks = _detection_ranks(["a2", "b1", "a1"], units)

    assert ranks == {"b1": 2, "a1": 3}


def test_curve_summary_includes_zero_budget() -> None:
    units = {"fault": frozenset(("bad",))}

    curve = _curve_summary([["bad", "ok"], ["ok", "bad"]], units, (0.0, 0.5, 1.0))

    assert curve["budget_cases"] == [0, 1, 2]
    assert curve["mean_faults"] == [0.0, 0.5, 1.0]


def test_fault_unit_apfd_uses_first_case_detecting_each_fault() -> None:
    units = {
        "fault-a": frozenset(("a-late", "a-early")),
        "fault-b": frozenset(("b",)),
    }

    score = _fault_unit_apfd(["a-early", "ok", "b", "a-late"], units)

    assert score == 0.625


def test_exclusive_summary_pairs_same_seed_discoveries() -> None:
    summary, probabilities = _exclusive_summary(
        {
            "candidate": [{"a", "b"}, {"a"}],
            "baseline": [{"b"}, {"c"}],
        },
        {"a", "b", "c"},
        candidate_arms=("candidate",),
        comparator_arms=("baseline",),
    )

    assert summary["candidate_only_faults"]["mean"] == 1.0
    assert summary["comparator_only_faults"]["mean"] == 0.5
    assert summary["overlapping_faults"]["mean"] == 0.5
    assert summary["robustly_exclusive_faults"] == ["a"]
    assert probabilities == {"a": 1.0, "b": 0.0, "c": 0.0}


def test_descending_score_orders_randomizes_only_ties() -> None:
    orders = _descending_score_orders(
        ("high", "low-a", "low-b"),
        {"high": 2, "low-a": 1, "low-b": 1},
        repeats=3,
        seed=42,
    )

    assert all(order[0] == "high" for order in orders)
    assert all(set(order[1:]) == {"low-a", "low-b"} for order in orders)


def test_coverage_curve_reports_observed_saturation() -> None:
    item_sets = {
        "both": frozenset(("a", "b")),
        "a": frozenset(("a",)),
        "b": frozenset(("b",)),
    }

    curve = _coverage_curve_summary(
        [["both", "a", "b"], ["a", "b", "both"]],
        item_sets,
        (0.0, 1 / 3, 1.0),
    )

    assert curve["n_observed_items"] == 2
    assert curve["mean_coverage"] == [0.0, 0.75, 1.0]
    assert curve["saturation_cases"] == {
        "median": 1.5,
        "p2_5": 1.025,
        "p97_5": 1.975,
        "min": 1,
        "max": 2,
    }


def test_concentration_reports_effective_support() -> None:
    result = _concentration(["a", "a", "b", "c"])

    assert result["n_unique_signatures"] == 3
    assert result["largest_signature_fraction"] == 0.5
    assert 2.0 < result["effective_number_of_signatures"] < 3.0


def test_transparent_keyword_coding() -> None:
    unsafe = {
        "planned_trajectory_safety_justification": (
            "Trajectory unsafe: the car drives onto the sidewalk."
        )
    }
    inconsistent = {
        "justification": [
            "Action inconsistent: trajectory changes left although CoT says keep lane."
        ]
    }

    assert _unsafe_mode(unsafe) == "road-edge or off-road excursion"
    assert _inconsistency_mode(inconsistent) == "lateral maneuver/direction mismatch"


def test_unsafe_mode_recognizes_opposing_and_stop_control() -> None:
    opposing = {
        "planned_trajectory_safety_justification": (
            "The trajectory crosses toward the opposing side of the road."
        )
    }
    stop_control = {
        "planned_trajectory_safety_justification": (
            "The ego does not stop for the stop sign before entering the intersection."
        )
    }

    assert _unsafe_mode(opposing) == "wrong-way or opposing-lane maneuver"
    assert _unsafe_mode(stop_control) == "traffic-control violation"


def test_trajectory_grid_item_parser() -> None:
    assert _trajectory_grid_cell("trajectory_grid:cell=4,-2") == (4, -2)


def test_cellwise_fault_values_average_positive_frames() -> None:
    result = _mean_fault_value_by_cell(
        {
            "fault-a": frozenset(((0, 0), (1, 0))),
            "fault-b": frozenset(((1, 0),)),
        },
        {"fault-a": 0.25, "fault-b": 0.75},
    )

    assert result == {
        (0, 0): {"n_positive_frames": 1, "mean_value": 0.25},
        (1, 0): {"n_positive_frames": 2, "mean_value": 0.5},
    }
