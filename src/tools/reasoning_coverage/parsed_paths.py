# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Coverage sets for LLM-extracted decision–role–proposition paths."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

ARM_NAMES = {
    "independent": "Independent (151)",
    "additive": "Fully additive (146)",
    "path_single": "Single-field paths",
    "path_le2": "Paths ≤2 fields",
    "path_all": "All partial paths",
    "z3_single": "Z3 values only (104)",
    "z3_all": "Z3 propositions only (16,073)",
}


def coverage_capacities(schema: dict[str, Any]) -> dict[str, int]:
    """Count the unpruned vocabulary and path universes, excluding missing values."""
    decision_variants = sum(
        math.prod(len(values) for values in decision["modifiers"].values())
        for decision in schema["decisions"].values()
    )
    decision_atoms = sum(
        (
            sum(len(values) for values in decision["modifiers"].values())
            if decision["modifiers"]
            else 1
        )
        for decision in schema["decisions"].values()
    )
    sizes = [
        [len(values) for values in fields.values()]
        for fields in schema["propositions"].values()
    ]
    singles = sum(sum(family) for family in sizes)
    pairs = sum(
        sum(a * b for a, b in itertools.combinations(family, 2)) for family in sizes
    )
    partials = sum(math.prod(size + 1 for size in family) - 1 for family in sizes)
    roles = len(schema["roles"])
    return {
        "independent": decision_variants + roles + singles,
        "additive": decision_atoms + roles + singles,
        "path_single": decision_variants * roles * singles,
        "path_le2": decision_variants * roles * (singles + pairs),
        "path_all": decision_variants * roles * partials,
        "z3_single": singles,
        "z3_all": partials,
    }


def _item(*parts: Any) -> str:
    return json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _valid_value(value: Any, vocabulary: list[Any]) -> bool:
    return any(
        type(value) is type(candidate) and value == candidate
        for candidate in vocabulary
    )


def parsed_coverage(
    record: dict[str, Any], schema: dict[str, Any]
) -> dict[str, frozenset[str]]:
    """Use all parsed mentions and explicit links, retaining empty coverage sets.

    Independent arms include unlinked mentions. Complete decision variants are
    required only for the independent-variant and path arms. The Z3 arms score
    propositions alone, so they need neither a complete decision nor a role
    link. Status, assertion, timing, and evidence do not add coverage dimensions
    or filter these arms.
    """
    if record["schema_version"] != schema["schema_version"]:
        raise ValueError(f"Unexpected parse version in {record['trace_id']}")
    result: dict[str, set[str]] = {arm: set() for arm in ARM_NAMES}
    decisions: dict[str, tuple[Any, ...] | None] = {}
    for decision in record["decisions"]:
        decision_id = decision["id"]
        if decision_id in decisions:
            raise ValueError(f"Duplicate decision ID: {decision_id}")
        family = decision["family"]
        definition = schema["decisions"][family]
        if decision["channel"] != definition["channel"]:
            raise ValueError(f"Wrong channel for decision {family}")
        modifiers = decision["modifiers"]
        if modifiers.keys() != definition["modifiers"].keys():
            raise ValueError(f"Wrong modifier keys for decision {family}")
        pairs = []
        for field, entry in sorted(modifiers.items()):
            value = entry["value"]
            if value is not None:
                if not _valid_value(value, definition["modifiers"][field]):
                    raise ValueError(
                        f"Invalid decision modifier: {family}.{field}={value!r}"
                    )
                result["additive"].add(
                    _item("decision_field", decision["channel"], family, field, value)
                )
            pairs.append((field, value))
        key = (decision["channel"], family, tuple(pairs))
        if not pairs:
            result["additive"].add(_item("decision", *key))
        if any(value is None for _, value in pairs):
            decisions[decision_id] = None
        else:
            decisions[decision_id] = key
            result["independent"].add(_item("decision", *key))

    propositions = {}
    for proposition in record["propositions"]:
        proposition_id = proposition["id"]
        if proposition_id in propositions:
            raise ValueError(f"Duplicate proposition ID: {proposition_id}")
        family = proposition["family"]
        definition = schema["propositions"][family]
        fields = []
        for field, entry in sorted(proposition["fields"].items()):
            value = entry["value"]
            if not _valid_value(value, definition[field]):
                raise ValueError(
                    f"Invalid proposition value: {family}.{field}={value!r}"
                )
            fields.append((field, value))
            item = _item("proposition_field", family, field, value)
            result["independent"].add(item)
            result["additive"].add(item)
        for size in range(1, len(fields) + 1):
            for subset in itertools.combinations(fields, size):
                item = _item("proposition", family, subset)
                result["z3_all"].add(item)
                if size == 1:
                    result["z3_single"].add(item)
        propositions[proposition_id] = (family, fields)

    for path in record["paths"]:
        role = path["role"]
        if role not in schema["roles"]:
            raise ValueError(f"Invalid role: {role}")
        decision = decisions[path["decision_id"]]
        family, fields = propositions[path["proposition_id"]]
        result["independent"].add(_item("role", role))
        result["additive"].add(_item("role", role))
        if decision is None:
            continue
        for size in range(1, len(fields) + 1):
            for subset in itertools.combinations(fields, size):
                item = _item("path", *decision, role, family, subset)
                result["path_all"].add(item)
                if size <= 2:
                    result["path_le2"].add(item)
                if size == 1:
                    result["path_single"].add(item)
    return {arm: frozenset(items) for arm, items in result.items()}


def load_parsed_coverage(
    input_dir: Path, output_dir: Path, schema: dict[str, Any]
) -> tuple[dict[str, dict[str, frozenset[str]]], dict[str, str]]:
    """Join parsed outputs to exact source text, checking evidence and ID coverage."""
    text_by_id = {}
    for path in sorted(input_dir.glob("*.json")):
        for record in json.loads(path.read_text()):
            trace_id = record["trace_id"]
            if trace_id in text_by_id:
                raise ValueError(f"Duplicate input trace: {trace_id}")
            text_by_id[trace_id] = record["reasoning"]
    if not text_by_id or len(set(text_by_id.values())) != len(text_by_id):
        raise ValueError("Expected nonempty, unique reasoning inputs")
    by_text = {}
    seen = set()
    for path in sorted(output_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            trace_id = record["trace_id"]
            if trace_id in seen:
                raise ValueError(f"Duplicate parsed trace: {trace_id}")
            text = text_by_id[trace_id]
            _check_evidence(record, text)
            by_text[text] = parsed_coverage(record, schema)
            seen.add(trace_id)
    if seen != text_by_id.keys():
        raise ValueError(
            f"Missing parser outputs: {sorted(text_by_id.keys() - seen)[:10]}"
        )
    return by_text, {text: trace_id for trace_id, text in text_by_id.items()}


def _check_evidence(value: Any, text: str) -> None:
    if isinstance(value, dict):
        if "evidence" in value:
            for quote in value["evidence"]:
                if not isinstance(quote, str) or not quote or quote not in text:
                    raise ValueError(
                        f"Evidence is not an exact text substring: {quote!r}"
                    )
        for child in value.values():
            _check_evidence(child, text)
    elif isinstance(value, list):
        for child in value:
            _check_evidence(child, text)
