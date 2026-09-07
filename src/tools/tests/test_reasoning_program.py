# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from reasoning_coverage.program import (
    compile_reasoning,
    cross_trace_transition_items,
    lexical_items,
    program_edge_items,
    program_node_items,
    program_path_items,
)


def test_compiler_does_not_infer_unstated_lateral_action() -> None:
    program = compile_reasoning(
        "Keep distance to the lead vehicle since it is moving slowly ahead"
    )
    step = program.steps[0]

    assert step.longitudinal_decision == "lead_obstacle_following"
    assert step.lateral_decision == "not_stated"
    assert step.lateral_action == "not_stated"
    assert {fact.predicate for fact in step.facts} >= {
        "lead_present",
        "lead_slow",
    }


def test_compiler_preserves_explicit_multi_step_program() -> None:
    program = compile_reasoning(
        "Change lanes to the right and decelerate for the turn, "
        "then accelerate once the path is clear"
    )

    assert len(program.steps) == 2
    assert program.steps[0].lateral_decision == "lane_change"
    assert program.steps[0].lateral_action == "steer_right"
    assert program.steps[0].longitudinal_action == "decelerate"
    assert program.steps[1].longitudinal_action == "accelerate"
    assert program.steps[1].entry_condition == "when_path_clear"
    assert any(
        item.startswith("program:temporal_long=")
        for item in program_path_items(program)
    )


def test_program_coverage_is_hierarchical_and_closed() -> None:
    program = compile_reasoning("Stop for the red traffic light at the intersection")

    nodes = program_node_items(program)
    edges = program_edge_items(program)
    paths = program_path_items(program)

    assert "program:fact=signal_stop" in nodes
    assert "program:objective=obey_signal_or_stop_control" in nodes
    assert "program:edge=signal_stop->obey_signal_or_stop_control" in edges
    assert any("signal_stop>obey_signal_or_stop_control" in item for item in paths)
    assert nodes <= edges <= paths


def test_cross_trace_transition_and_lexical_baseline() -> None:
    previous = compile_reasoning("Keep lane since the lane is clear ahead")
    current = compile_reasoning("Nudge left due to cones blocking the right side")

    transitions = cross_trace_transition_items(previous, current)

    assert any(item.startswith("program:trace_transition_lat=") for item in transitions)
    assert "text:bigram=nudge_left" in lexical_items(current)
