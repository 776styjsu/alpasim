# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import copy
import hashlib
import json

import pytest
from reasoning_coverage.experiment_analysis import DatasetAnalysis
from reasoning_coverage.parsed_evaluation import _load_grid, _unique_discovery
from reasoning_coverage.parsed_paths import (
    ARM_NAMES,
    load_parsed_coverage,
    parsed_coverage,
)
from reasoning_coverage.trajectory_grid import trajectory_grid_items


@pytest.fixture
def schema():
    return {
        "schema_version": "test",
        "decisions": {
            "lane_keeping": {"channel": "lateral", "modifiers": {}},
            "lane_change": {
                "channel": "lateral",
                "modifiers": {"direction": ["left", "right"]},
            },
        },
        "roles": ["motivates", "blocks"],
        "propositions": {
            "critical_object": {
                "type": ["vehicle", "pedestrian"],
                "in_path": [True, False],
                "motion": ["stopped", "slowing"],
            }
        },
    }


@pytest.fixture
def record():
    return {
        "schema_version": "test",
        "trace_id": "trace",
        "decisions": [
            {
                "id": "d1",
                "channel": "lateral",
                "family": "lane_keeping",
                "modifiers": {},
            }
        ],
        "propositions": [
            {
                "id": "p1",
                "family": "critical_object",
                "fields": {
                    "type": {"value": "vehicle"},
                    "in_path": {"value": False},
                    "motion": {"value": "stopped"},
                },
            }
        ],
        "paths": [{"decision_id": "d1", "proposition_id": "p1", "role": "motivates"}],
    }


def test_field_subsets_deduplicate_repeated_paths_and_keep_false(schema, record):
    record["paths"] *= 2
    items = parsed_coverage(record, schema)
    assert len(items["path_single"]) == 3
    assert len(items["path_le2"]) == 6
    assert len(items["path_all"]) == 7
    decoded = [json.loads(item) for item in items["path_single"]]
    assert any(item[-1] == [["in_path", False]] for item in decoded)
    record["propositions"][0]["fields"]["in_path"]["value"] = 0
    with pytest.raises(ValueError, match="Invalid proposition value"):
        parsed_coverage(record, schema)


def test_partial_decision_loses_path_credit_but_not_independent_mentions(
    schema, record
):
    record["decisions"].append(
        {
            "id": "d2",
            "channel": "lateral",
            "family": "lane_change",
            "modifiers": {"direction": {"value": None}},
        }
    )
    record["paths"].append(
        {"decision_id": "d2", "proposition_id": "p1", "role": "blocks"}
    )
    incomplete = parsed_coverage(record, schema)
    assert len(incomplete["path_all"]) == 7
    assert any(
        json.loads(item) == ["role", "blocks"] for item in incomplete["independent"]
    )
    record["decisions"][1]["modifiers"]["direction"]["value"] = "left"
    complete = parsed_coverage(record, schema)
    assert len(complete["path_all"]) == 14
    assert len(complete["independent"] - incomplete["independent"]) == 1
    assert len(complete["additive"] - incomplete["additive"]) == 1


def test_unlinked_propositions_do_not_create_paths_or_cross_entity_pairs(
    schema, record
):
    before = parsed_coverage(record, schema)
    record["propositions"].append(
        {
            "id": "p2",
            "family": "critical_object",
            "fields": {"type": {"value": "pedestrian"}},
        }
    )
    after = parsed_coverage(record, schema)
    assert after["path_all"] == before["path_all"]
    assert len(after["independent"] - before["independent"]) == 1
    record["paths"].append(
        {"decision_id": "d1", "proposition_id": "p2", "role": "motivates"}
    )
    linked = parsed_coverage(record, schema)
    assert len(linked["path_all"] - before["path_all"]) == 1


def test_z3_arms_score_propositions_without_decision_or_role(schema, record):
    record["decisions"] = []
    record["paths"] = []
    items = parsed_coverage(record, schema)
    assert not items["path_all"]
    assert len(items["z3_single"]) == 3
    assert len(items["z3_all"]) == 7
    record["propositions"].append(
        {
            "id": "p2",
            "family": "critical_object",
            "fields": {"type": {"value": "pedestrian"}},
        }
    )
    extended = parsed_coverage(record, schema)
    assert len(extended["z3_single"] - items["z3_single"]) == 1
    assert len(extended["z3_all"] - items["z3_all"]) == 1


def test_empty_extractions_remain_empty_and_metadata_does_not_change_items(
    schema, record
):
    before = parsed_coverage(record, schema)
    changed = copy.deepcopy(record)
    changed["decisions"][0].update(status="blocked", timing="continuation")
    changed["propositions"][0]["assertion"] = "uncertain"
    changed["diagnostics"] = [{"detail": "test"}]
    assert parsed_coverage(changed, schema) == before
    record.update(decisions=[], propositions=[], paths=[])
    assert all(not items for items in parsed_coverage(record, schema).values())


def test_parse_loader_requires_exact_evidence_and_all_ids(tmp_path, schema, record):
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    (inputs / "batch.json").write_text(
        json.dumps([{"trace_id": "trace", "reasoning": "Stopped vehicle."}])
    )
    record["paths"][0]["evidence"] = ["Invented text"]
    output = outputs / "batch.jsonl"
    output.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="Evidence is not an exact"):
        load_parsed_coverage(inputs, outputs, schema)
    record["paths"][0]["evidence"] = ["Stopped vehicle"]
    output.write_text(json.dumps(record) + "\n")
    parsed, ids = load_parsed_coverage(inputs, outputs, schema)
    assert ids == {"Stopped vehicle.": "trace"}
    assert parsed["Stopped vehicle."] == parsed_coverage(record, schema)
    output.write_text("")
    with pytest.raises(ValueError, match="Missing parser outputs"):
        load_parsed_coverage(inputs, outputs, schema)


def test_grid_loading_matches_existing_control_and_rejects_missing_data(tmp_path):
    metadata = tmp_path / "metadata.json"
    info = tmp_path / "additional_info.json"
    payload = {
        "coordinate_frame": "ego_rig_frame",
        "trajectory_source": "trajectory_xy_rig_frame",
        "waypoints": [
            {"relative_time_s": 0, "position_m": {"x": 0, "y": 0}},
            {"relative_time_s": 3, "position_m": {"x": 4, "y": 0}},
        ],
    }
    info.write_text(json.dumps(payload))
    items, manifest = _load_grid(metadata)
    assert items == trajectory_grid_items(metadata)
    assert len(items) == 3
    assert manifest["sha256"] == hashlib.sha256(info.read_bytes()).hexdigest()
    info.unlink()
    with pytest.raises(FileNotFoundError):
        _load_grid(metadata)


def test_unique_discovery_reports_equal_budget_overlap_and_both_directions():
    case_ids = tuple(["a", "b", "c", *[f"ok{i}" for i in range(17)]])
    candidate = ["a", "b", *[key for key in case_ids if key not in ("a", "b")]]
    grid = ["b", "c", *[key for key in case_ids if key not in ("b", "c")]]
    orders = {arm: [candidate, candidate] for arm in ARM_NAMES}
    orders["trajectory_grid"] = [grid, grid]
    units = {key: frozenset((key,)) for key in ("a", "b", "c")}
    analysis = DatasetAnalysis(case_ids, {}, orders, {"collision": units})
    report, rows, faults = _unique_discovery(analysis, {key: {} for key in units})
    comparison = report["collision"]["0.1"]["independent"]
    assert (
        comparison["candidate_detected"]["mean"]
        == comparison["grid_detected"]["mean"]
        == 2
    )
    assert (
        comparison["candidate_only_faults"]["mean"]
        == comparison["comparator_only_faults"]["mean"]
        == 1
    )
    assert comparison["overlapping_faults"]["mean"] == 1
    assert comparison["robustly_exclusive_faults"] == ["a"]
    assert comparison["robustly_grid_exclusive_faults"] == ["c"]
    assert len(rows) == 4 * len(ARM_NAMES)
    assert len(faults) == 3 * len(rows)
