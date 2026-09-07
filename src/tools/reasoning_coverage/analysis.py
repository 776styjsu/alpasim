# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Reasoning-coverage extraction and test-suite prioritization.

The pilot intentionally operates on already executed benchmark cases. It tests
whether a coverage abstraction retains failure-revealing cases under a smaller
review/test-suite budget; it does not claim to avoid the cost of obtaining the
reasoning trace in the first place.
"""

from __future__ import annotations

import ast
import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .program import (
    compile_reasoning,
    lexical_items,
    program_edge_items,
    program_node_items,
    program_path_items,
)
from .trajectory_grid import (
    DEFAULT_CELL_SIZE_M,
    DEFAULT_HORIZON_S,
    DEFAULT_RASTER_STEP_M,
    trajectory_grid_items,
)

SCENE_FIELDS = (
    "behavior",
    "layout",
    "lighting",
    "road_types",
    "surface_conditions",
    "traffic_density",
    "weather",
    "vrus",
)
TRAJECTORY_FEATURES = (
    "total_distance_2d_m",
    "final_lateral_offset_m",
    "mean_speed_mps",
    "max_abs_longitudinal_acceleration_mps2",
    "heading_change_deg",
    "max_abs_curvature_1pm",
)
VISUAL_FEATURES = (
    "mean_luma",
    "std_luma",
    "entropy",
    "edge_strength",
    "colorfulness",
)
IMAGE_GRID_SIZE = 8
IMAGE_GRID_FEATURES = tuple(
    f"grid_{row}_{column}_{channel}"
    for row in range(IMAGE_GRID_SIZE)
    for column in range(IMAGE_GRID_SIZE)
    for channel in ("r", "g", "b")
)
OUTCOMES = (
    "unreliable_cot",
    "inconsistent_reliable_cot",
    "unsafe_trajectory",
    "any_issue",
)
ARMS = (
    "random",
    "specification",
    "visual_stats",
    "image_grid",
    "trajectory",
    "trajectory_grid",
    "reasoning_node",
    "reasoning_graph",
    "reasoning_fine",
    "text_bigram",
    "program_node",
    "program_edge",
    "program_path",
    "hybrid",
    "program_hybrid",
)


def _tuple_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    return tuple(str(item) for item in values if item not in (None, ""))


def _flatten_cot(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    if not isinstance(value, str):
        return "" if value is None else str(value)
    stripped = value.strip()
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return stripped
    if isinstance(parsed, list):
        return " | ".join(str(item) for item in parsed)
    return str(parsed)


def _reliability_flag(entry: dict[str, Any]) -> bool | None:
    nested = entry.get("cot_reliability")
    value = (
        nested.get("reliable")
        if isinstance(nested, dict) and "reliable" in nested
        else entry.get("cot_reliable")
    )
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "reliable", "yes", "1"}:
            return True
        if normalized in {"false", "unreliable", "no", "0"}:
            return False
    return None


@dataclass(frozen=True)
class ReasoningState:
    """Canonical text-derived Alpamayo reasoning state for one decision."""

    longitudinal_decision: str
    lateral_decision: str
    causal_categories: tuple[str, ...]
    critical_tags: tuple[str, ...]
    longitudinal_actions: tuple[str, ...]
    lateral_actions: tuple[str, ...]

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> "ReasoningState":
        clip_id = entry.get("clip_id", "<unknown>")
        try:
            label = entry["labels"]["cot_decision_label"]
            decision = label["high_level_decision"]
            actions = label["atomic_meta_action_hint"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing reasoning label for {clip_id}") from exc

        longitudinal_actions = _tuple_values(
            actions.get("longitudinal_sequence") or actions.get("longitudinal")
        )
        lateral_actions = _tuple_values(
            actions.get("lateral_sequence") or actions.get("lateral")
        )
        return cls(
            longitudinal_decision=str(decision["longitudinal"]),
            lateral_decision=str(decision["lateral"]),
            causal_categories=tuple(
                sorted(_tuple_values(label.get("critical_component_categories")))
            ),
            critical_tags=tuple(
                sorted(_tuple_values(label.get("critical_component_tags")))
            ),
            longitudinal_actions=longitudinal_actions,
            lateral_actions=lateral_actions,
        )


@dataclass(frozen=True)
class BenchmarkCase:
    """All coverage inputs and independent outcomes for one benchmark case."""

    clip_id: str
    chain_of_thought: str
    reasoning: ReasoningState
    reasoning_label_source: str
    scene_fields: tuple[tuple[str, tuple[str, ...]], ...]
    trajectory_values: tuple[tuple[str, float], ...]
    trajectory_categories: tuple[tuple[str, str], ...]
    visual_values: tuple[tuple[str, float], ...]
    outcomes: tuple[tuple[str, bool], ...]
    trajectory_grid_items: tuple[str, ...] = ()

    def outcome(self, name: str) -> bool:
        return dict(self.outcomes)[name]


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("results", payload.get("entries"))
    else:
        entries = None
    if not isinstance(entries, list):
        raise ValueError(f"Expected a benchmark list or results/entries object: {path}")
    return [entry for entry in entries if isinstance(entry, dict)]


def apply_label_audit(
    entries: list[dict[str, Any]], audit_path: Path | None
) -> tuple[list[dict[str, Any]], int]:
    """Overlay reviewed reasoning labels while retaining all outcome annotations."""
    if audit_path is None:
        return entries, 0
    with audit_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    annotations = payload.get("annotations") if isinstance(payload, dict) else None
    if not isinstance(annotations, list):
        raise ValueError(f"Audit file has no annotations list: {audit_path}")
    by_clip: dict[str, dict[str, Any]] = {}
    for item in annotations:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("clip_id"), str)
            or not isinstance(item.get("reasoning_label"), dict)
        ):
            raise ValueError(f"Malformed audit annotation in {audit_path}")
        clip_id = item["clip_id"]
        if clip_id in by_clip:
            raise ValueError(f"Duplicate audit annotation for {clip_id}")
        by_clip[clip_id] = item["reasoning_label"]

    overlaid: list[dict[str, Any]] = []
    applied = 0
    for original in entries:
        entry = copy.deepcopy(original)
        replacement = by_clip.get(entry.get("clip_id"))
        if replacement is not None:
            entry.setdefault("labels", {})["cot_decision_label"] = copy.deepcopy(
                replacement
            )
            applied += 1
        overlaid.append(entry)
    unapplied = set(by_clip) - {entry.get("clip_id") for entry in entries}
    if unapplied:
        raise ValueError(f"Audit IDs absent from benchmark: {sorted(unapplied)}")
    return overlaid, applied


def _resolve_metadata_path(entry: dict[str, Any], benchmark_path: Path) -> Path | None:
    raw_path = entry.get("source_scene_file")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else []
    root = benchmark_path.resolve().parent
    candidates.extend((root / path, root / "tutorial_alpamayo" / path))
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _trajectory_features(
    metadata_path: Path | None,
) -> tuple[dict[str, float], dict[str, str]]:
    if metadata_path is None:
        return {}, {}
    info_path = metadata_path.with_name("additional_info.json")
    try:
        with info_path.open("r", encoding="utf-8") as file:
            info = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}, {}

    summary = info.get("summary", {})
    continuous: dict[str, float] = {}
    for name in TRAJECTORY_FEATURES:
        value = _safe_float(summary.get(name))
        if value is None:
            continue
        if name in {
            "final_lateral_offset_m",
            "heading_change_deg",
            "max_abs_curvature_1pm",
        }:
            value = abs(value)
        continuous[name] = value

    dominant = info.get("meta_actions", {}).get("dominant", {})
    categorical = {
        f"dominant_{axis}": str(dominant[axis])
        for axis in ("longitudinal", "lateral")
        if dominant.get(axis)
    }
    return continuous, categorical


def _visual_features(metadata_path: Path | None) -> dict[str, float]:
    if metadata_path is None:
        return {}
    image_path = metadata_path.with_name("camera_front_wide_120fov.jpg")
    try:
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32)
    except (FileNotFoundError, OSError):
        return {}

    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    histogram, _ = np.histogram(luma, bins=32, range=(0.0, 256.0))
    probabilities = histogram / max(1, histogram.sum())
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    horizontal_edges = np.abs(np.diff(luma, axis=1)).mean()
    vertical_edges = np.abs(np.diff(luma, axis=0)).mean()
    colorfulness = (rgb.max(axis=2) - rgb.min(axis=2)).mean()
    features = {
        "mean_luma": float(luma.mean()),
        "std_luma": float(luma.std()),
        "entropy": entropy,
        "edge_strength": float((horizontal_edges + vertical_edges) / 2.0),
        "colorfulness": float(colorfulness),
    }
    grid = np.asarray(
        Image.fromarray(rgb.astype(np.uint8)).resize(
            (IMAGE_GRID_SIZE, IMAGE_GRID_SIZE), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    for row in range(IMAGE_GRID_SIZE):
        for column in range(IMAGE_GRID_SIZE):
            for channel_index, channel in enumerate(("r", "g", "b")):
                features[f"grid_{row}_{column}_{channel}"] = float(
                    grid[row, column, channel_index]
                )
    return features


def _outcomes(entry: dict[str, Any]) -> dict[str, bool]:
    reliable = _reliability_flag(entry)
    inconsistent_reliable = (
        reliable is True and entry.get("cot_action_consistency") is False
    )
    values = {
        "unreliable_cot": reliable is False,
        "inconsistent_reliable_cot": inconsistent_reliable,
        "unsafe_trajectory": entry.get("planned_trajectory_safe") is False,
    }
    values["any_issue"] = any(values.values())
    return values


def build_cases(
    benchmark_path: Path, audit_path: Path | None = None
) -> tuple[list[BenchmarkCase], dict[str, int]]:
    entries, audited = apply_label_audit(load_benchmark(benchmark_path), audit_path)
    cases: list[BenchmarkCase] = []
    artifact_counts = {
        "entries": len(entries),
        "audited_reasoning_labels": audited,
        "trajectory_features": 0,
        "trajectory_grid": 0,
        "visual_features": 0,
    }
    seen: set[str] = set()
    for entry in entries:
        clip_id = str(entry.get("clip_id", "")).strip()
        if not clip_id:
            raise ValueError("Benchmark entry has no clip_id")
        if clip_id in seen:
            raise ValueError(f"Duplicate clip_id: {clip_id}")
        seen.add(clip_id)

        metadata_path = _resolve_metadata_path(entry, benchmark_path)
        trajectory_values, trajectory_categories = _trajectory_features(metadata_path)
        grid_items = trajectory_grid_items(metadata_path)
        visual_values = _visual_features(metadata_path)
        artifact_counts["trajectory_features"] += bool(trajectory_values)
        artifact_counts["trajectory_grid"] += bool(grid_items)
        artifact_counts["visual_features"] += bool(visual_values)

        scene = entry.get("labels", {}).get("nurec_scene", {})
        scene_fields = tuple(
            (field, _tuple_values(scene.get(field)))
            for field in SCENE_FIELDS
            if _tuple_values(scene.get(field))
        )
        label = entry["labels"]["cot_decision_label"]
        cases.append(
            BenchmarkCase(
                clip_id=clip_id,
                chain_of_thought=_flatten_cot(entry.get("chain_of_thought")),
                reasoning=ReasoningState.from_entry(entry),
                reasoning_label_source=str(label.get("source", "unknown")),
                scene_fields=scene_fields,
                trajectory_values=tuple(sorted(trajectory_values.items())),
                trajectory_categories=tuple(sorted(trajectory_categories.items())),
                visual_values=tuple(sorted(visual_values.items())),
                outcomes=tuple(sorted(_outcomes(entry).items())),
                trajectory_grid_items=tuple(sorted(grid_items)),
            )
        )
    return cases, artifact_counts


@dataclass(frozen=True)
class QuantileDiscretizer:
    edges: dict[str, tuple[float, ...]]

    @classmethod
    def fit(
        cls, rows: Iterable[dict[str, float]], feature_names: Iterable[str], bins: int
    ) -> "QuantileDiscretizer":
        if bins < 2:
            raise ValueError("bins must be at least 2")
        materialized = list(rows)
        edges: dict[str, tuple[float, ...]] = {}
        for name in feature_names:
            values = [row[name] for row in materialized if name in row]
            if not values:
                continue
            raw_edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
            edges[name] = tuple(sorted(set(float(edge) for edge in raw_edges)))
        return cls(edges=edges)

    def label(self, feature: str, value: float) -> str:
        edges = self.edges[feature]
        return f"q{int(np.searchsorted(edges, value, side='right'))}"


def _reasoning_node_items(state: ReasoningState) -> set[str]:
    items = {
        f"reasoning:longitudinal={state.longitudinal_decision}",
        f"reasoning:lateral={state.lateral_decision}",
    }
    items.update(f"reasoning:cause={cause}" for cause in state.causal_categories)
    return items


def _reasoning_graph_items(state: ReasoningState) -> set[str]:
    items = _reasoning_node_items(state)
    items.add(
        "reasoning:decision_pair="
        f"{state.longitudinal_decision}|{state.lateral_decision}"
    )
    for cause in state.causal_categories:
        items.add(f"reasoning:cause_to_long={cause}->{state.longitudinal_decision}")
        items.add(f"reasoning:cause_to_lat={cause}->{state.lateral_decision}")
    return items


def _reasoning_fine_items(state: ReasoningState) -> set[str]:
    items = _reasoning_graph_items(state)
    items.update(f"reasoning:tag={tag}" for tag in state.critical_tags)
    items.update(
        f"reasoning:longitudinal_action={action}"
        for action in state.longitudinal_actions
    )
    items.update(
        f"reasoning:lateral_action={action}" for action in state.lateral_actions
    )
    return items


def _specification_items(case: BenchmarkCase) -> set[str]:
    atomic: list[tuple[str, str]] = []
    for field, values in case.scene_fields:
        atomic.extend((field, value) for value in values)
    items = {f"spec:{field}={value}" for field, value in atomic}
    for left, right in combinations(atomic, 2):
        if left[0] == right[0]:
            continue
        pair = sorted((f"{left[0]}={left[1]}", f"{right[0]}={right[1]}"))
        items.add(f"spec:pair={pair[0]}|{pair[1]}")
    return items


def _continuous_items(
    namespace: str,
    values: tuple[tuple[str, float], ...],
    discretizer: QuantileDiscretizer,
) -> set[str]:
    return {
        f"{namespace}:{name}={discretizer.label(name, value)}"
        for name, value in values
        if name in discretizer.edges
    }


def build_coverage_sets(
    cases: list[BenchmarkCase], bins: int = 4
) -> tuple[dict[str, dict[str, frozenset[str]]], dict[str, Any]]:
    trajectory_discretizer = QuantileDiscretizer.fit(
        (dict(case.trajectory_values) for case in cases), TRAJECTORY_FEATURES, bins
    )
    visual_discretizer = QuantileDiscretizer.fit(
        (dict(case.visual_values) for case in cases),
        (*VISUAL_FEATURES, *IMAGE_GRID_FEATURES),
        bins,
    )

    coverage: dict[str, dict[str, frozenset[str]]] = {
        arm: {} for arm in ARMS if arm != "random"
    }
    for case in cases:
        reasoning_node = _reasoning_node_items(case.reasoning)
        reasoning_graph = _reasoning_graph_items(case.reasoning)
        reasoning_fine = _reasoning_fine_items(case.reasoning)
        program = compile_reasoning(case.chain_of_thought)
        text_bigram = lexical_items(program)
        program_node = program_node_items(program)
        program_edge = program_edge_items(program)
        program_path = program_path_items(program)
        specification = _specification_items(case)
        trajectory = _continuous_items(
            "trajectory", case.trajectory_values, trajectory_discretizer
        )
        trajectory.update(
            f"trajectory:{name}={value}" for name, value in case.trajectory_categories
        )
        categories = dict(case.trajectory_categories)
        if {"dominant_longitudinal", "dominant_lateral"} <= categories.keys():
            trajectory.add(
                "trajectory:action_pair="
                f"{categories['dominant_longitudinal']}|{categories['dominant_lateral']}"
            )
        visual = _continuous_items("visual", case.visual_values, visual_discretizer)
        visual_stats = {
            item
            for item in visual
            if item.removeprefix("visual:").split("=", maxsplit=1)[0] in VISUAL_FEATURES
        }
        image_grid = visual - visual_stats
        hybrid = specification | trajectory | reasoning_graph
        program_hybrid = specification | trajectory | program_path

        values = {
            "specification": specification,
            "visual_stats": visual_stats,
            "image_grid": image_grid,
            "trajectory": trajectory,
            "trajectory_grid": set(case.trajectory_grid_items),
            "reasoning_node": reasoning_node,
            "reasoning_graph": reasoning_graph,
            "reasoning_fine": reasoning_fine,
            "text_bigram": text_bigram,
            "program_node": program_node,
            "program_edge": program_edge,
            "program_path": program_path,
            "hybrid": hybrid,
            "program_hybrid": program_hybrid,
        }
        for arm, items in values.items():
            coverage[arm][case.clip_id] = frozenset(items)

    metadata = {
        "bins": bins,
        "trajectory_grid": {
            "coordinate_frame": "ego_rig_frame",
            "cell_size_m": DEFAULT_CELL_SIZE_M,
            "horizon_s": DEFAULT_HORIZON_S,
            "raster_step_m": DEFAULT_RASTER_STEP_M,
            "origin_centered_cells": True,
        },
        "trajectory_quantile_edges": trajectory_discretizer.edges,
        "visual_quantile_edges": visual_discretizer.edges,
    }
    return coverage, metadata


def greedy_coverage_order(
    case_ids: Iterable[str], item_sets: dict[str, frozenset[str]], seed: int
) -> list[str]:
    """Order cases by maximum new coverage, randomizing only equal-gain ties."""
    remaining = list(case_ids)
    random.Random(seed).shuffle(remaining)
    tie_order = {clip_id: index for index, clip_id in enumerate(remaining)}
    covered: set[str] = set()
    selected: list[str] = []
    while remaining:
        clip_id = max(
            remaining,
            key=lambda candidate: (
                len(item_sets[candidate] - covered),
                -tie_order[candidate],
            ),
        )
        remaining.remove(clip_id)
        selected.append(clip_id)
        covered.update(item_sets[clip_id])
    return selected


def apfd(order: list[str], failing_ids: set[str]) -> float | None:
    """Average Percentage of Faults Detected, treating each failing case as a fault."""
    if not failing_ids:
        return None
    ranks = {clip_id: rank for rank, clip_id in enumerate(order, start=1)}
    missing = failing_ids - ranks.keys()
    if missing:
        raise ValueError(f"Failing IDs absent from ordering: {sorted(missing)}")
    n = len(order)
    return (
        1.0
        - sum(ranks[clip_id] for clip_id in failing_ids) / (n * len(failing_ids))
        + 1.0 / (2.0 * n)
    )


def _recall_curve(order: list[str], failing_ids: set[str]) -> list[float]:
    if not failing_ids:
        return [0.0] * len(order)
    found = 0
    curve: list[float] = []
    for clip_id in order:
        found += clip_id in failing_ids
        curve.append(found / len(failing_ids))
    return curve


def _coverage_curve(
    order: list[str], item_sets: dict[str, frozenset[str]]
) -> list[float]:
    universe = set().union(*(item_sets[clip_id] for clip_id in order))
    if not universe:
        return [0.0] * len(order)
    covered: set[str] = set()
    curve = []
    for clip_id in order:
        covered.update(item_sets[clip_id])
        curve.append(len(covered) / len(universe))
    return curve


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = np.asarray(values, dtype=float)
    return {
        "mean": float(fmean(values)),
        "std": float(pstdev(values)),
        "p2_5": float(np.quantile(ordered, 0.025)),
        "p97_5": float(np.quantile(ordered, 0.975)),
    }


def _mean_curve(curves: list[list[float]]) -> list[float]:
    return np.asarray(curves, dtype=float).mean(axis=0).tolist()


def _budget_index(size: int, fraction: float) -> int:
    return max(0, min(size - 1, math.ceil(size * fraction) - 1))


def _summarize_arm(
    arm: str,
    cases: list[BenchmarkCase],
    item_sets: dict[str, frozenset[str]] | None,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    case_ids = sorted(case.clip_id for case in cases)
    failing = {
        outcome: {case.clip_id for case in cases if case.outcome(outcome)}
        for outcome in OUTCOMES
    }
    orders: list[list[str]] = []
    for repeat in range(repeats):
        repeat_seed = seed + repeat
        if arm == "random":
            order = list(case_ids)
            random.Random(repeat_seed).shuffle(order)
        else:
            assert item_sets is not None
            order = greedy_coverage_order(case_ids, item_sets, repeat_seed)
        orders.append(order)

    outcome_summaries: dict[str, Any] = {}
    for outcome in OUTCOMES:
        curves = [_recall_curve(order, failing[outcome]) for order in orders]
        apfd_values = [apfd(order, failing[outcome]) for order in orders]
        apfd_present = [value for value in apfd_values if value is not None]
        mean_curve = _mean_curve(curves)
        outcome_summaries[outcome] = {
            "n_failures": len(failing[outcome]),
            "apfd": _distribution(apfd_present) if apfd_present else None,
            "mean_recall_at_budget": {
                str(fraction): mean_curve[_budget_index(len(cases), fraction)]
                for fraction in (0.1, 0.25, 0.5)
            },
            "mean_recall_curve": mean_curve,
        }

    if item_sets is None:
        coverage_summary = None
    else:
        universe = set().union(*item_sets.values())
        coverage_curves = [_coverage_curve(order, item_sets) for order in orders]
        mean_coverage_curve = _mean_curve(coverage_curves)
        coverage_summary = {
            "n_items": len(universe),
            "mean_recall_at_budget": {
                str(fraction): mean_coverage_curve[_budget_index(len(cases), fraction)]
                for fraction in (0.1, 0.25, 0.5)
            },
            "mean_recall_curve": mean_coverage_curve,
        }

    return {
        "repeats": repeats,
        "representative_order": orders[0],
        "coverage": coverage_summary,
        "outcomes": outcome_summaries,
    }


def run_pilot(
    cases: list[BenchmarkCase], bins: int = 4, repeats: int = 200, seed: int = 42
) -> dict[str, Any]:
    if not cases:
        raise ValueError("No benchmark cases")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    coverage, discretization = build_coverage_sets(cases, bins=bins)
    label_sources: dict[str, int] = {}
    for case in cases:
        label_sources[case.reasoning_label_source] = (
            label_sources.get(case.reasoning_label_source, 0) + 1
        )

    return {
        "schema_version": 1,
        "kind": "alpamayo_reasoning_coverage_pilot",
        "limitations": [
            "Retrospective suite prioritization; coverage is observed after execution.",
            "The development benchmark was itself sampled partly for semantic coverage.",
            "visual_stats is a low-cost image-statistics baseline, not neuron coverage.",
            "image_grid is quantized 8x8 RGB input coverage, not full-resolution pixel or activation coverage.",
            "trajectory_grid is ego-centric plan-shape coverage, not global-map coverage.",
            "Intervals reflect randomized tie breaking, not independent test corpora.",
        ],
        "config": {"bins": bins, "repeats": repeats, "seed": seed},
        "dataset": {
            "n_cases": len(cases),
            "reasoning_label_sources": label_sources,
            "outcome_counts": {
                outcome: sum(case.outcome(outcome) for case in cases)
                for outcome in OUTCOMES
            },
        },
        "discretization": discretization,
        "arms": {
            arm: _summarize_arm(
                arm,
                cases,
                coverage.get(arm),
                repeats=repeats,
                seed=seed,
            )
            for arm in ARMS
        },
    }


def write_curve_csv(report: dict[str, Any], path: Path) -> None:
    arms = report["arms"]
    n_cases = report["dataset"]["n_cases"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["arm", "budget", "budget_fraction", "coverage_recall", *OUTCOMES]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for arm, summary in arms.items():
            coverage = summary["coverage"]
            for index in range(n_cases):
                writer.writerow(
                    {
                        "arm": arm,
                        "budget": index + 1,
                        "budget_fraction": (index + 1) / n_cases,
                        "coverage_recall": (
                            coverage["mean_recall_curve"][index]
                            if coverage is not None
                            else ""
                        ),
                        **{
                            outcome: summary["outcomes"][outcome]["mean_recall_curve"][
                                index
                            ]
                            for outcome in OUTCOMES
                        },
                    }
                )
