# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json
from pathlib import Path

import pytest
from reasoning_coverage.alpamayo_tables import (
    table1_decision_pairs,
    table2_category_x_table1_decision_pairs,
    table5_action_pairs,
)
from reasoning_coverage.table_program import (
    action_items,
    atomic_reasoning_path_items,
    compile_table_program,
    compress_action_pairs,
    load_action_runs,
    primary_reasoning_items,
    reasoning_action_items,
    sequence_reasoning_items,
    z1_category_x_z3_pair_items,
    z3_pair_items,
)


def test_table_program_compiles_components_and_closed_decision() -> None:
    program = compile_table_program(
        "Keep distance to the slow lead vehicle directly ahead in our lane"
    )

    assert program.primary.decision.longitudinal == "lead_obstacle_following"
    assert program.primary.decision.lateral == "none"
    assert {component.category for component in program.primary.components} == {
        "critical_objects"
    }
    assert {component.predicate for component in program.primary.components} >= {
        "lead_present",
        "lead_slow",
    }
    assert all(clause.role == "proximate_cause" for clause in program.primary.clauses)


def test_explicit_continuation_is_reasoning_but_not_primary() -> None:
    program = compile_table_program(
        "Change lanes to the left due to a slow lead vehicle, "
        "then match speed once established in the target lane"
    )

    assert len(program.stages) == 2
    assert program.primary.decision.lateral == "lane_change"
    assert program.primary.decision.lateral_direction == "left"
    assert program.continuations[0].decision.longitudinal == "set_speed_tracking"
    assert program.continuations[0].temporal_guard == "after_previous_complete"

    primary = primary_reasoning_items(program)
    sequence = sequence_reasoning_items(program)
    assert "table:z3_longitudinal=set_speed_tracking" not in primary
    assert "table:z3_longitudinal=set_speed_tracking" in sequence
    assert any(item.startswith("table:transition=") for item in sequence)
    assert primary <= sequence


def test_atomic_reasoning_paths_keep_only_complete_z1_z2_z3_links() -> None:
    program = compile_table_program(
        "Change lanes left due to a slow lead vehicle and a clear gap"
    )

    paths = atomic_reasoning_path_items(program)

    assert paths
    assert all(item.startswith("table:path=") for item in paths)
    assert all(item.count(">") == 2 for item in paths)
    assert {
        "table:path=critical_objects/lead_slow>proximate_cause>lateral:lane_change",
        "table:path=lane_lanelines/gap_available>enabling_condition>lateral:lane_change",
    } <= paths


def test_action_is_separate_and_uses_full_compressed_sequence() -> None:
    runs = compress_action_pairs(
        (
            ("maintain_speed", "go_straight"),
            ("gentle_accelerate", "steer_left"),
            ("gentle_accelerate", "steer_left"),
            ("maintain_speed", "go_straight"),
        )
    )
    assert [run.segment_count for run in runs] == [1, 2, 1]

    program = compile_table_program("Change lanes left due to a clear gap")
    reasoning = sequence_reasoning_items(program)
    actions = action_items(runs)
    combined = reasoning_action_items(program, runs)

    assert not any(item.startswith("table:a_") for item in reasoning)
    assert "table:a_pair=gentle_accelerate|steer_left" in actions
    assert any(item.startswith("table:a_transition=") for item in actions)
    assert reasoning | actions <= combined
    assert any(item.startswith("table:z3a_pair=") for item in combined)


def test_load_action_runs_validates_and_orders_segments(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    payload = {
        "meta_actions": {
            "per_segment": [
                {
                    "index": 1,
                    "longitudinal": "gentle_decelerate",
                    "lateral": "steer_right",
                },
                {
                    "index": 0,
                    "longitudinal": "maintain_speed",
                    "lateral": "go_straight",
                },
            ]
        }
    }
    (tmp_path / "additional_info.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    runs = load_action_runs(metadata_path)

    assert [run.pair for run in runs] == [
        "maintain_speed|go_straight",
        "gentle_decelerate|steer_right",
    ]


def test_published_closed_denominators_have_expected_sizes() -> None:
    assert len(table1_decision_pairs()) == 71
    assert len(table2_category_x_table1_decision_pairs()) == 497
    assert len(table5_action_pairs()) == 49


def test_z3_and_z1_x_z3_are_pure_pair_ablation_items() -> None:
    program = compile_table_program(
        "Change lanes left due to a slow lead vehicle, then maintain speed"
    )

    assert z3_pair_items(program) == {
        "table:z3_pair=lead_obstacle_following|lane_change"
    }
    assert z3_pair_items(program, include_continuations=True) == {
        "table:z3_pair=lead_obstacle_following|lane_change",
        "table:z3_pair=set_speed_tracking|none",
    }
    primary_products = z1_category_x_z3_pair_items(program)
    assert primary_products == {
        "table:z1_category_x_z3_pair=critical_objects|"
        "lead_obstacle_following|lane_change"
    }
    assert all("z2" not in item for item in primary_products)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Maintain speed", "set_speed_tracking"),
        ("Slow for a lead vehicle", "lead_obstacle_following"),
        ("Slow for the road curve", "speed_adaptation_road_events"),
        (
            "Search for a usable gap before merging",
            "gap_searching_for_lane_change_merge_zipper",
        ),
        ("Accelerate to pass the stopped car", "acceleration_for_passing_overtaking"),
        ("Yield to the pedestrian", "yield_agent_right_of_way"),
        ("Stop for a red traffic light", "stop_for_static_constraints"),
    ],
)
def test_each_table1_longitudinal_decision_is_reachable(
    text: str, expected: str
) -> None:
    assert compile_table_program(text).primary.decision.longitudinal == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Keep the lane centered", "lane_keeping_centering"),
        ("Take the off-ramp", "merge_split_facility_change"),
        ("Nudge left out-of-lane", "out_of_lane_nudge"),
        ("Nudge right within the lane", "in_lane_nudge"),
        ("Change lanes left", "lane_change"),
        ("Pull over right", "pull_over_curb_approach"),
        ("Turn left through the intersection", "turn"),
        ("Abort the lane change", "lateral_maneuver_abort"),
    ],
)
def test_each_table1_lateral_decision_is_reachable(text: str, expected: str) -> None:
    assert compile_table_program(text).primary.decision.lateral == expected
