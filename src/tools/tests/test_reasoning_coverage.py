# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json
from pathlib import Path

import pytest
from reasoning_coverage.analysis import (
    BenchmarkCase,
    ReasoningState,
    apfd,
    apply_label_audit,
    build_cases,
    build_coverage_sets,
    greedy_coverage_order,
    load_benchmark,
)


def _case(clip_id: str, state: ReasoningState) -> BenchmarkCase:
    return BenchmarkCase(
        clip_id=clip_id,
        chain_of_thought="test reasoning",
        reasoning=state,
        reasoning_label_source="test",
        scene_fields=(("weather", ("clear",)),),
        trajectory_values=(("mean_speed_mps", 4.0),),
        trajectory_categories=(("dominant_lateral", "go_straight"),),
        visual_values=(("mean_luma", 100.0),),
        outcomes=(("any_issue", False),),
    )


def test_reasoning_graph_encodes_nodes_pair_and_causal_edges() -> None:
    state = ReasoningState(
        longitudinal_decision="yield_agent_right_of_way",
        lateral_decision="turn",
        causal_categories=("critical_objects", "yield_stop_control"),
        critical_tags=("pedestrian",),
        longitudinal_actions=("gentle_decelerate",),
        lateral_actions=("steer_right",),
    )

    coverage, _ = build_coverage_sets([_case("case-1", state)])

    graph = coverage["reasoning_graph"]["case-1"]
    assert "reasoning:longitudinal=yield_agent_right_of_way" in graph
    assert "reasoning:lateral=turn" in graph
    assert "reasoning:decision_pair=yield_agent_right_of_way|turn" in graph
    assert "reasoning:cause_to_lat=critical_objects->turn" in graph
    assert "reasoning:tag=pedestrian" not in graph
    assert "reasoning:tag=pedestrian" in coverage["reasoning_fine"]["case-1"]


def test_greedy_order_maximizes_new_items_and_uses_seed_for_ties() -> None:
    item_sets = {
        "broad": frozenset({"a", "b"}),
        "a-only": frozenset({"a"}),
        "c-only": frozenset({"c"}),
    }

    order = greedy_coverage_order(item_sets, item_sets, seed=7)

    assert order[0] == "broad"
    assert order[1] == "c-only"
    assert set(order) == set(item_sets)


def test_apfd_uses_failure_rank() -> None:
    order = ["failure-first", "pass", "failure-last"]

    score = apfd(order, {"failure-first", "failure-last"})

    assert score == pytest.approx(0.5)
    with pytest.raises(ValueError, match="absent"):
        apfd(order, {"missing"})


def test_label_audit_overlays_only_reasoning_label(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    original_label = {"source": "heuristic", "high_level_decision": {}}
    entry = {
        "clip_id": "case-1",
        "labels": {
            "cot_decision_label": original_label,
            "nurec_scene": {"weather": ["clear"]},
        },
        "planned_trajectory_safe": False,
    }
    benchmark_path.write_text(json.dumps([entry]), encoding="utf-8")
    replacement = {
        "source": "audit",
        "high_level_decision": {
            "longitudinal": "none",
            "lateral": "lane_keeping_centering",
        },
        "atomic_meta_action_hint": {
            "longitudinal_sequence": [],
            "lateral_sequence": ["go_straight"],
        },
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {"annotations": [{"clip_id": "case-1", "reasoning_label": replacement}]}
        ),
        encoding="utf-8",
    )

    entries = load_benchmark(benchmark_path)
    overlaid, count = apply_label_audit(entries, audit_path)

    assert count == 1
    assert overlaid[0]["labels"]["cot_decision_label"] == replacement
    assert overlaid[0]["labels"]["nurec_scene"] == {"weather": ["clear"]}
    assert overlaid[0]["planned_trajectory_safe"] is False
    assert entries[0]["labels"]["cot_decision_label"] == original_label


def test_label_audit_rejects_unknown_clip_id(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "annotations": [
                    {"clip_id": "unknown", "reasoning_label": {"source": "audit"}}
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absent from benchmark"):
        apply_label_audit([{"clip_id": "known"}], audit_path)


def test_duplicate_clip_ids_fail_fast(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    label = {
        "source": "test",
        "high_level_decision": {
            "longitudinal": "none",
            "lateral": "lane_keeping_centering",
        },
        "atomic_meta_action_hint": {
            "longitudinal_sequence": [],
            "lateral_sequence": ["go_straight"],
        },
    }
    benchmark_path.write_text(
        json.dumps(
            [
                {"clip_id": "same", "labels": {"cot_decision_label": label}},
                {"clip_id": "same", "labels": {"cot_decision_label": label}},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate clip_id"):
        build_cases(benchmark_path)
