# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""OOD hazard-suite evaluation for closed reasoning-program coverage."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

import numpy as np

from .analysis import (
    IMAGE_GRID_FEATURES,
    TRAJECTORY_FEATURES,
    VISUAL_FEATURES,
    _trajectory_features,
    _visual_features,
    apfd,
)
from .program import (
    ReasoningProgram,
    compile_reasoning,
    cross_trace_transition_items,
    lexical_items,
    program_edge_items,
    program_node_items,
    program_path_items,
)
from .statistics import (
    paired_apfd_deltas,
    permutation_test_apfd_delta,
    source_group_bootstrap_apfd_delta,
)
from .trajectory_grid import (
    DEFAULT_CELL_SIZE_M,
    DEFAULT_HORIZON_S,
    DEFAULT_RASTER_STEP_M,
    trajectory_grid_items,
)

OOD_ARMS = (
    "random",
    "visual_stats",
    "image_grid",
    "trajectory",
    "trajectory_grid",
    "text_bigram",
    "text_bigram_support2",
    "program_node",
    "program_node_support2",
    "program_edge",
    "program_edge_support2",
    "program_path",
    "program_path_support2",
    "program_trace",
    "program_trace_support2",
    "program_trace_support5",
    "hybrid",
)
DERIVED_SUPPORT_ARMS = {
    "text_bigram_support2",
    "program_node_support2",
    "program_edge_support2",
    "program_path_support2",
    "program_trace_support2",
    "program_trace_support5",
}
SUPPORT_ARM_SPECS = (
    ("text_bigram_support2", "text_bigram", 2),
    ("program_node_support2", "program_node", 2),
    ("program_edge_support2", "program_edge", 2),
    ("program_path_support2", "program_path", 2),
    ("program_trace_support2", "program_trace", 2),
    ("program_trace_support5", "program_trace", 5),
)
OUTCOME_NAMES = (
    "closed_loop_failure",
    "manual_unsafe_scene",
    "closed_loop_safety_event",
)
FEATURE_CACHE_VERSION = 2


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    scene_id: str
    metadata_path: Path
    program: ReasoningProgram
    trajectory_values: tuple[tuple[str, float], ...]
    trajectory_categories: tuple[tuple[str, str], ...]
    visual_values: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    scene_path: Path
    frames: tuple[FrameRecord, ...]
    outcomes: tuple[tuple[str, bool], ...]

    def outcome(self, name: str) -> bool:
        return dict(self.outcomes)[name]


def _base_clip_id(clip_id: str) -> str:
    return clip_id.rsplit("_t", maxsplit=1)[0]


def _timestep(frame_id: str) -> int:
    try:
        return int(frame_id.rsplit("_t", maxsplit=1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Frame ID has no numeric timestep: {frame_id}") from exc


def _scene_path(metadata_path: Path, root: Path) -> Path:
    relative = metadata_path.relative_to(root)
    return root / relative.parts[0]


def _load_entries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("entries", payload.get("results"))
    else:
        entries = None
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        raise ValueError(f"Expected a list of entry objects: {path}")
    return entries


def _development_scene_ids(benchmark_path: Path) -> set[str]:
    return {
        _base_clip_id(str(entry["clip_id"])) for entry in _load_entries(benchmark_path)
    }


def _manual_unsafe_ids(unsafe_path: Path) -> tuple[set[str], set[str]]:
    entries = _load_entries(unsafe_path)
    return (
        {str(entry["clip_id"]) for entry in entries},
        {_base_clip_id(str(entry["clip_id"])) for entry in entries},
    )


def _scene_outcomes(
    scene_path: Path,
    manual_unsafe_scenes: set[str],
    expected_scene_id: str,
) -> dict[str, bool] | None:
    summary_path = scene_path / "aggregate" / "results-summary.json"
    try:
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    rollouts = summary.get("rollouts")
    metrics_results = summary.get("metrics_results")
    if not isinstance(rollouts, list) or len(rollouts) != 1:
        raise ValueError(f"Expected one rollout in {summary_path}")
    if not isinstance(metrics_results, list) or len(metrics_results) != 1:
        raise ValueError(f"Expected one metric result in {summary_path}")
    rollout = rollouts[0]
    metrics = metrics_results[0]
    scene_id = str(rollout["clipgt_id"])
    if scene_id != expected_scene_id:
        raise ValueError(
            f"Scene ID mismatch in {summary_path}: "
            f"expected {expected_scene_id}, got {scene_id}"
        )
    passed = rollout.get("passed")
    if not isinstance(passed, bool):
        raise ValueError(f"Expected Boolean passed in {summary_path}")
    safety_metric_names = (
        "collision_at_fault",
        "offroad",
        "wrong_lane",
        "safety_monitor_triggered",
    )
    safety_metric_values: list[float] = []
    for name in safety_metric_names:
        if name not in metrics:
            raise ValueError(f"Missing safety metric {name} in {summary_path}")
        try:
            value = float(metrics[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Safety metric {name} is not numeric in {summary_path}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"Safety metric {name} is not finite in {summary_path}")
        safety_metric_values.append(value)
    safety_event = any(value > 0.0 for value in safety_metric_values)
    return {
        "closed_loop_failure": not passed,
        "manual_unsafe_scene": scene_id in manual_unsafe_scenes,
        "closed_loop_safety_event": safety_event,
    }


def _cache_fingerprint(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(FEATURE_CACHE_VERSION).encode())
    digest.update(str(root.resolve()).encode())
    digest.update(_file_sha256(Path(__file__).with_name("analysis.py")).encode())
    for metadata_path in paths:
        for path in (
            metadata_path,
            metadata_path.with_name("additional_info.json"),
            metadata_path.with_name("camera_front_wide_120fov.jpg"),
        ):
            try:
                stat = path.stat()
            except FileNotFoundError as exc:
                raise ValueError(f"Missing OOD feature input: {path}") from exc
            digest.update(str(path.relative_to(root)).encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _file_collection_sha256(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_one(metadata_path: Path) -> tuple[Any, ...]:
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    frame_id = metadata_path.parent.name
    scene_id = _base_clip_id(str(metadata["clip_id"]))
    trajectory_values, trajectory_categories = _trajectory_features(metadata_path)
    visual_values = _visual_features(metadata_path)
    return (
        frame_id,
        scene_id,
        metadata.get("chain_of_thought", ""),
        trajectory_values,
        trajectory_categories,
        visual_values,
    )


def _write_feature_cache(
    cache_path: Path,
    fingerprint: str,
    paths: list[Path],
    extracted: list[tuple[Any, ...]],
) -> None:
    payload = {
        "schema_version": FEATURE_CACHE_VERSION,
        "fingerprint": fingerprint,
        "metadata_paths": [str(path) for path in paths],
        "frames": [
            {
                "frame_id": frame_id,
                "scene_id": scene_id,
                "chain_of_thought": cot,
                "trajectory_values": trajectory_values,
                "trajectory_categories": trajectory_categories,
                "visual_values": visual_values,
            }
            for (
                frame_id,
                scene_id,
                cot,
                trajectory_values,
                trajectory_categories,
                visual_values,
            ) in extracted
        ],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))


def _load_or_extract_frames(
    root: Path, cache_path: Path, workers: int
) -> tuple[list[FrameRecord], dict[str, Any]]:
    paths = sorted(root.rglob("metadata.json"), key=str)
    if not paths:
        raise ValueError(f"No metadata.json files under {root}")
    fingerprint = _cache_fingerprint(paths, root)
    cache_hit = False
    payload: dict[str, Any] | None = None
    try:
        with cache_path.open("r", encoding="utf-8") as file:
            candidate = json.load(file)
        if (
            candidate.get("schema_version") == FEATURE_CACHE_VERSION
            and candidate.get("fingerprint") == fingerprint
            and candidate.get("metadata_paths") == [str(path) for path in paths]
            and isinstance(candidate.get("frames"), list)
            and len(candidate["frames"]) == len(paths)
        ):
            payload = candidate
            cache_hit = True
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass

    if payload is None:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            extracted = list(executor.map(_extract_one, paths))
        _write_feature_cache(cache_path, fingerprint, paths, extracted)
        raw_frames = [
            {
                "frame_id": frame_id,
                "scene_id": scene_id,
                "chain_of_thought": cot,
                "trajectory_values": trajectory_values,
                "trajectory_categories": trajectory_categories,
                "visual_values": visual_values,
            }
            for (
                frame_id,
                scene_id,
                cot,
                trajectory_values,
                trajectory_categories,
                visual_values,
            ) in extracted
        ]
    else:
        raw_frames = payload["frames"]
        paths = [Path(path) for path in payload["metadata_paths"]]

    program_cache: dict[str, ReasoningProgram] = {}
    frames: list[FrameRecord] = []
    for path, raw in zip(paths, raw_frames, strict=True):
        cot = str(raw["chain_of_thought"])
        program = program_cache.get(cot)
        if program is None:
            program = compile_reasoning(cot)
            program_cache[cot] = program
        frames.append(
            FrameRecord(
                frame_id=str(raw["frame_id"]),
                scene_id=str(raw["scene_id"]),
                metadata_path=path,
                program=program,
                trajectory_values=tuple(
                    sorted(
                        (str(name), float(value))
                        for name, value in raw["trajectory_values"].items()
                    )
                ),
                trajectory_categories=tuple(
                    sorted(
                        (str(name), str(value))
                        for name, value in raw["trajectory_categories"].items()
                    )
                ),
                visual_values=tuple(
                    sorted(
                        (str(name), float(value))
                        for name, value in raw["visual_values"].items()
                    )
                ),
            )
        )
    return frames, {
        "cache_path": str(cache_path),
        "cache_hit": cache_hit,
        "fingerprint": fingerprint,
        "n_metadata_files": len(paths),
        "n_unique_cot_strings": len(program_cache),
    }


def load_ood_cases(
    root: Path,
    benchmark_path: Path,
    unsafe_path: Path,
    cache_path: Path,
    workers: int,
) -> tuple[list[FrameRecord], list[SceneRecord], dict[str, Any]]:
    frames, cache_info = _load_or_extract_frames(root, cache_path, workers)
    development_scenes = _development_scene_ids(benchmark_path)
    manual_unsafe_frames, manual_unsafe_scenes = _manual_unsafe_ids(unsafe_path)

    grouped: dict[str, list[FrameRecord]] = defaultdict(list)
    scene_paths: dict[str, Path] = {}
    frame_paths: dict[str, Path] = {}
    for frame in frames:
        existing_frame_path = frame_paths.setdefault(
            frame.frame_id, frame.metadata_path
        )
        if existing_frame_path != frame.metadata_path:
            raise ValueError(
                f"Duplicate frame ID {frame.frame_id}: "
                f"{existing_frame_path} and {frame.metadata_path}"
            )
        grouped[frame.scene_id].append(frame)
        current_scene_path = _scene_path(frame.metadata_path, root)
        existing_scene_path = scene_paths.setdefault(frame.scene_id, current_scene_path)
        if existing_scene_path != current_scene_path:
            raise ValueError(
                f"Scene ID {frame.scene_id} appears under both "
                f"{existing_scene_path} and {current_scene_path}"
            )
    for values in grouped.values():
        values.sort(key=lambda frame: _timestep(frame.frame_id))

    strict_frames = [
        frame for frame in frames if frame.scene_id not in development_scenes
    ]
    scenes: list[SceneRecord] = []
    for scene_id, scene_frames in sorted(grouped.items()):
        if scene_id in development_scenes:
            continue
        outcomes = _scene_outcomes(
            scene_paths[scene_id], manual_unsafe_scenes, scene_id
        )
        if outcomes is None:
            continue
        scenes.append(
            SceneRecord(
                scene_id=scene_id,
                scene_path=scene_paths[scene_id],
                frames=tuple(scene_frames),
                outcomes=tuple(sorted(outcomes.items())),
            )
        )

    strict_scene_ids = {scene.scene_id for scene in scenes}
    manual_strict_frames = manual_unsafe_frames & {
        frame.frame_id for frame in strict_frames
    }
    audit = {
        **cache_info,
        "n_all_frames": len(frames),
        "n_all_scenes": len(grouped),
        "n_development_overlap_scenes": len(set(grouped) & development_scenes),
        "n_strict_frames": len(strict_frames),
        "n_strict_scenes_with_outcomes": len(scenes),
        "n_strict_scene_frames": sum(len(scene.frames) for scene in scenes),
        "n_manual_unsafe_frames_strict": len(manual_strict_frames),
        "n_manual_unsafe_scenes_with_outcomes": sum(
            scene.outcome("manual_unsafe_scene") for scene in scenes
        ),
        "strict_scene_ids": sorted(strict_scene_ids),
    }
    return strict_frames, scenes, audit


def _quantile_edges(
    records: Iterable[FrameRecord],
    attribute: str,
    feature_names: Iterable[str],
    bins: int,
) -> dict[str, tuple[float, ...]]:
    columns: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for name, value in getattr(record, attribute):
            columns[name].append(value)
    edges: dict[str, tuple[float, ...]] = {}
    for name in feature_names:
        values = columns[name]
        if values:
            quantiles = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
            edges[name] = tuple(sorted(set(float(value) for value in quantiles)))
    return edges


def _continuous_items(
    namespace: str,
    values: tuple[tuple[str, float], ...],
    edges: dict[str, tuple[float, ...]],
) -> set[str]:
    return {
        sys.intern(
            f"{namespace}:{name}=q"
            f"{int(np.searchsorted(edges[name], value, side='right'))}"
        )
        for name, value in values
        if name in edges
    }


def build_frame_coverage(
    frames: list[FrameRecord], bins: int
) -> tuple[dict[str, dict[str, frozenset[str]]], dict[str, Any]]:
    trajectory_edges = _quantile_edges(
        frames, "trajectory_values", TRAJECTORY_FEATURES, bins
    )
    visual_edges = _quantile_edges(
        frames,
        "visual_values",
        (*VISUAL_FEATURES, *IMAGE_GRID_FEATURES),
        bins,
    )
    coverage = {
        arm: {}
        for arm in OOD_ARMS
        if arm != "random" and arm not in DERIVED_SUPPORT_ARMS
    }
    previous_by_scene: dict[str, ReasoningProgram] = {}
    for frame in sorted(
        frames, key=lambda item: (item.scene_id, _timestep(item.frame_id))
    ):
        nodes = {sys.intern(item) for item in program_node_items(frame.program)}
        edges = {sys.intern(item) for item in program_edge_items(frame.program)}
        paths = {sys.intern(item) for item in program_path_items(frame.program)}
        trace = set(paths)
        previous = previous_by_scene.get(frame.scene_id)
        if previous is not None:
            trace.update(
                sys.intern(item)
                for item in cross_trace_transition_items(previous, frame.program)
            )
        previous_by_scene[frame.scene_id] = frame.program

        text = {sys.intern(item) for item in lexical_items(frame.program)}
        trajectory = _continuous_items(
            "trajectory", frame.trajectory_values, trajectory_edges
        )
        trajectory.update(
            sys.intern(f"trajectory:{name}={value}")
            for name, value in frame.trajectory_categories
        )
        categories = dict(frame.trajectory_categories)
        if {"dominant_longitudinal", "dominant_lateral"} <= categories.keys():
            trajectory.add(
                sys.intern(
                    "trajectory:action_pair="
                    f"{categories['dominant_longitudinal']}|"
                    f"{categories['dominant_lateral']}"
                )
            )
        visual = _continuous_items("visual", frame.visual_values, visual_edges)
        visual_stats = {
            item
            for item in visual
            if item.removeprefix("visual:").split("=", maxsplit=1)[0] in VISUAL_FEATURES
        }
        image_grid = visual - visual_stats
        values = {
            "visual_stats": visual_stats,
            "image_grid": image_grid,
            "trajectory": trajectory,
            "trajectory_grid": trajectory_grid_items(frame.metadata_path),
            "text_bigram": text,
            "program_node": nodes,
            "program_edge": edges,
            "program_path": paths,
            "program_trace": trace,
            "hybrid": trajectory | trace,
        }
        for arm, items in values.items():
            coverage[arm][frame.frame_id] = frozenset(items)

    support_by_source = {
        source: Counter(
            item for item_set in coverage[source].values() for item in item_set
        )
        for source in (
            "text_bigram",
            "program_node",
            "program_edge",
            "program_path",
            "program_trace",
        )
    }
    for target, source, minimum in SUPPORT_ARM_SPECS:
        support = support_by_source[source]
        coverage[target] = {
            frame_id: frozenset(item for item in items if support[item] >= minimum)
            for frame_id, items in coverage[source].items()
        }
    return coverage, {
        "bins": bins,
        "trajectory_grid": {
            "coordinate_frame": "ego_rig_frame",
            "cell_size_m": DEFAULT_CELL_SIZE_M,
            "horizon_s": DEFAULT_HORIZON_S,
            "raster_step_m": DEFAULT_RASTER_STEP_M,
            "origin_centered_cells": True,
        },
        "trajectory_quantile_edges": trajectory_edges,
        "visual_quantile_edges": visual_edges,
        "singleton_items": {
            source: sum(value == 1 for value in support.values())
            for source, support in support_by_source.items()
        },
    }


def aggregate_scene_coverage(
    scenes: list[SceneRecord],
    frame_coverage: dict[str, dict[str, frozenset[str]]],
) -> dict[str, dict[str, frozenset[str]]]:
    coverage: dict[str, dict[str, frozenset[str]]] = {arm: {} for arm in frame_coverage}
    for scene in scenes:
        for arm, item_sets in frame_coverage.items():
            coverage[arm][scene.scene_id] = frozenset().union(
                *(item_sets[frame.frame_id] for frame in scene.frames)
            )

    for target, source, minimum in SUPPORT_ARM_SPECS:
        support = Counter(
            item for item_set in coverage[source].values() for item in item_set
        )
        coverage[target] = {
            scene_id: frozenset(item for item in items if support[item] >= minimum)
            for scene_id, items in coverage[source].items()
        }
    return coverage


def lazy_greedy_coverage_order(
    case_ids: Iterable[str],
    item_sets: dict[str, frozenset[str]],
    seed: int,
    *,
    multi_round: bool = False,
) -> list[str]:
    """Order cases by greedy marginal coverage gain with randomized tie breaks.

    Single-round ordering stops once no remaining case adds an uncovered item and
    appends the rest in seeded random order. Multi-round ordering instead clears
    the covered set and keeps selecting, so every position stays
    coverage-directed: round ``r`` covers each observed item for the ``r``-th
    time. Without it, an arm whose universe saturates early is compared mostly on
    its random tail rather than on its abstraction.

    Cases sharing an item set have equal gain at every step, so the search ranks
    distinct item sets and draws each group's members in seeded tie order. The
    corpora repeat item sets heavily, and multi-round ordering has to rank every
    case rather than stopping at saturation, so this grouping is what keeps the
    cost workable.
    """
    case_ids = list(case_ids)
    random.Random(seed).shuffle(case_ids)
    groups: dict[frozenset[str], list[str]] = defaultdict(list)
    for case_id in case_ids:
        groups[item_sets[case_id]].append(case_id)
    signatures = list(groups)
    members = [groups[signature] for signature in signatures]
    # Members are in shuffled order, so a group's best tie break is its next one.
    tie_order = {case_id: index for index, case_id in enumerate(case_ids)}
    taken = [0] * len(signatures)

    def entry(index: int, gain: int) -> tuple[int, int, int]:
        return -gain, tie_order[members[index][taken[index]]], index

    covered: set[str] = set()
    selected: list[str] = []
    heap = [entry(index, len(signatures[index])) for index in range(len(signatures))]
    heapq.heapify(heap)
    while heap:
        negative_stored_gain, _, index = heapq.heappop(heap)
        if taken[index] == len(members[index]):
            continue
        actual_gain = len(signatures[index] - covered)
        if actual_gain != -negative_stored_gain:
            heapq.heappush(heap, entry(index, actual_gain))
            continue
        if actual_gain == 0:
            if not multi_round or not covered:
                break
            covered = set()
            heap = [
                entry(other, len(signatures[other]))
                for other in range(len(signatures))
                if taken[other] < len(members[other])
            ]
            heapq.heapify(heap)
            continue
        selected.append(members[index][taken[index]])
        taken[index] += 1
        covered.update(signatures[index])
        if taken[index] < len(members[index]):
            heapq.heappush(heap, entry(index, len(signatures[index] - covered)))
    chosen = set(selected)
    selected.extend(case_id for case_id in case_ids if case_id not in chosen)
    return selected


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(fmean(values)),
        "std": float(pstdev(values)),
        "p2_5": float(np.quantile(array, 0.025)),
        "p97_5": float(np.quantile(array, 0.975)),
    }


def _orders(
    case_ids: list[str],
    coverage: dict[str, dict[str, frozenset[str]]],
    repeats: int,
    seed: int,
) -> dict[str, list[list[str]]]:
    result: dict[str, list[list[str]]] = {arm: [] for arm in OOD_ARMS}
    ordered_ids = sorted(case_ids)
    for repeat in range(repeats):
        repeat_seed = seed + repeat
        random_order = list(ordered_ids)
        random.Random(repeat_seed).shuffle(random_order)
        result["random"].append(random_order)
        for arm in OOD_ARMS:
            if arm == "random":
                continue
            result[arm].append(
                lazy_greedy_coverage_order(ordered_ids, coverage[arm], repeat_seed)
            )
    return result


def _recall(order: list[str], failures: set[str], budget: int) -> float:
    return len(set(order[:budget]) & failures) / len(failures)


def evaluate_orders(
    orders: dict[str, list[list[str]]],
    failures_by_outcome: dict[str, set[str]],
    outcome_label_scope: dict[str, str],
    source_group_by_case: dict[str, str],
    permutation_count: int,
    bootstrap_count: int,
    seed: int,
) -> dict[str, Any]:
    n_cases = len(next(iter(orders.values()))[0])
    outcomes: dict[str, Any] = {}
    for outcome, failures in failures_by_outcome.items():
        if not failures:
            continue
        arm_results: dict[str, Any] = {}
        for arm, arm_orders in orders.items():
            apfd_values = [
                float(apfd(order, failures))
                for order in arm_orders
                if apfd(order, failures) is not None
            ]
            arm_results[arm] = {
                "apfd": _distribution(apfd_values),
                "mean_recall_at_budget": {
                    str(fraction): float(
                        fmean(
                            _recall(
                                order,
                                failures,
                                max(1, math.ceil(n_cases * fraction)),
                            )
                            for order in arm_orders
                        )
                    )
                    for fraction in (0.05, 0.1, 0.25, 0.5)
                },
            }
        label_scope = outcome_label_scope[outcome]
        if label_scope not in {"exhaustive", "positive_unlabeled"}:
            raise ValueError(f"Unknown outcome label scope: {label_scope}")
        comparisons_by_baseline: dict[str, dict[str, Any]] = {}
        for baseline in ("random", "trajectory"):
            comparisons: dict[str, Any] = {}
            for arm in orders:
                if arm == baseline:
                    continue
                paired = paired_apfd_deltas(orders[arm], orders[baseline], failures)
                comparison: dict[str, Any] = {
                    "paired_apfd_delta": paired["delta"],
                }
                if label_scope == "exhaustive":
                    comparison["permutation"] = permutation_test_apfd_delta(
                        orders[arm],
                        orders[baseline],
                        failures,
                        permutations=permutation_count,
                        seed=seed,
                    )
                    comparison["source_group_bootstrap"] = (
                        source_group_bootstrap_apfd_delta(
                            orders[arm],
                            orders[baseline],
                            failures,
                            source_group_by_case,
                            resamples=bootstrap_count,
                            seed=seed,
                        )
                    )
                comparisons[arm] = comparison
            comparisons_by_baseline[baseline] = comparisons
        outcomes[outcome] = {
            "n_failures": len(failures),
            "label_scope": label_scope,
            "arms": arm_results,
            "comparisons_to_random": comparisons_by_baseline["random"],
            "comparisons_to_trajectory": comparisons_by_baseline["trajectory"],
        }
    return outcomes


def _coverage_metadata(
    coverage: dict[str, dict[str, frozenset[str]]],
) -> dict[str, Any]:
    return {
        arm: {
            "n_items": len(set().union(*item_sets.values())),
            "mean_items_per_case": float(
                fmean(len(items) for items in item_sets.values())
            ),
        }
        for arm, item_sets in coverage.items()
    }


def run_ood_evaluation(
    root: Path,
    benchmark_path: Path,
    unsafe_path: Path,
    cache_path: Path,
    *,
    bins: int,
    scene_repeats: int,
    frame_repeats: int,
    permutations: int,
    bootstraps: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    strict_frames, scenes, dataset_audit = load_ood_cases(
        root, benchmark_path, unsafe_path, cache_path, workers
    )
    scene_frames = [frame for scene in scenes for frame in scene.frames]
    scene_frame_coverage, scene_discretization = build_frame_coverage(
        scene_frames, bins
    )
    scene_coverage = aggregate_scene_coverage(scenes, scene_frame_coverage)
    frame_coverage, frame_discretization = build_frame_coverage(strict_frames, bins)

    scene_ids = [scene.scene_id for scene in scenes]
    scene_orders = _orders(scene_ids, scene_coverage, scene_repeats, seed)
    scene_failures = {
        outcome: {scene.scene_id for scene in scenes if scene.outcome(outcome)}
        for outcome in OUTCOME_NAMES
    }
    scene_results = evaluate_orders(
        scene_orders,
        scene_failures,
        {
            "closed_loop_failure": "exhaustive",
            "manual_unsafe_scene": "positive_unlabeled",
            "closed_loop_safety_event": "exhaustive",
        },
        {scene_id: scene_id for scene_id in scene_ids},
        permutations,
        bootstraps,
        seed,
    )

    unsafe_frame_ids, _ = _manual_unsafe_ids(unsafe_path)
    strict_frame_ids = [frame.frame_id for frame in strict_frames]
    known_unsafe = unsafe_frame_ids & set(strict_frame_ids)
    frame_orders = _orders(strict_frame_ids, frame_coverage, frame_repeats, seed)
    frame_results = evaluate_orders(
        frame_orders,
        {"known_manual_unsafe_event": known_unsafe},
        {"known_manual_unsafe_event": "positive_unlabeled"},
        {frame.frame_id: frame.scene_id for frame in strict_frames},
        permutations,
        bootstraps,
        seed,
    )

    unique_programs = {frame.program.text: frame.program for frame in strict_frames}
    program_diagnostics = {
        "n_unique_programs": len(unique_programs),
        "n_program_steps": sum(
            len(program.steps) for program in unique_programs.values()
        ),
        "n_multistep_programs": sum(
            len(program.steps) > 1 for program in unique_programs.values()
        ),
        "ambiguity_status": dict(
            Counter(program.ambiguity_status for program in unique_programs.values())
        ),
    }

    return {
        "schema_version": 2,
        "kind": "reasoning_program_ood_hazard_evaluation",
        "inputs": {
            "root": str(root),
            "development_benchmark": str(benchmark_path),
            "development_benchmark_sha256": _file_sha256(benchmark_path),
            "manual_unsafe_events": str(unsafe_path),
            "manual_unsafe_events_sha256": _file_sha256(unsafe_path),
            "ood_feature_dependency_fingerprint": dataset_audit["fingerprint"],
            "outcome_summaries_sha256": _file_collection_sha256(
                (
                    scene.scene_path / "aggregate" / "results-summary.json"
                    for scene in scenes
                ),
                root,
            ),
            "program_compiler_sha256": _file_sha256(
                Path(__file__).with_name("program.py")
            ),
            "feature_extractor_sha256": _file_sha256(
                Path(__file__).with_name("analysis.py")
            ),
            "statistics_sha256": _file_sha256(
                Path(__file__).with_name("statistics.py")
            ),
            "evaluator_sha256": _file_sha256(Path(__file__)),
        },
        "config": {
            "bins": bins,
            "scene_repeats": scene_repeats,
            "frame_repeats": frame_repeats,
            "permutations": permutations,
            "bootstraps": bootstraps,
            "seed": seed,
            "workers": workers,
        },
        "dataset_audit": dataset_audit,
        "program_diagnostics": program_diagnostics,
        "discretization": {
            "scene_level": scene_discretization,
            "frame_level_challenge": frame_discretization,
        },
        "limitations": [
            "The OOD suite and unsafe examples were inspected before this exploratory run.",
            "Manual unsafe labels are positive-unlabeled; inferential tests are not run for them.",
            "Closed-loop failure and unsafe planned-timestep labels are different outcomes.",
            "The deterministic text compiler is consistent but not fully human validated.",
            "Scene coverage uses a full same-run trace and may include post-outcome states.",
            "Quantile bins are fit transductively on each OOD cohort without outcome access.",
            "Bootstrap intervals condition on fixed coverage orderings.",
            "Arms and outcomes are unadjusted for multiplicity in this adaptive analysis.",
            "Greedy union coverage weights every emitted item equally across feature families.",
            "Feature-cache fingerprints use paths, byte sizes, and modification times rather than full image hashes.",
        ],
        "scene_level": {
            "n_cases": len(scene_ids),
            "outcome_counts": {
                name: len(values) for name, values in scene_failures.items()
            },
            "coverage": _coverage_metadata(scene_coverage),
            "outcomes": scene_results,
            "representative_orders": {
                arm: arm_orders[0] for arm, arm_orders in scene_orders.items()
            },
        },
        "frame_level_challenge": {
            "n_cases": len(strict_frame_ids),
            "n_known_manual_unsafe_events": len(known_unsafe),
            "coverage": _coverage_metadata(frame_coverage),
            "outcomes": frame_results,
            "representative_orders": {
                arm: arm_orders[0] for arm, arm_orders in frame_orders.items()
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate closed reasoning-program coverage on OOD hazards"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/project/AV/drive-the-thoughts/alpasim/tutorial_alpamayo_ood_hazard_2604"
        ),
    )
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark.json"))
    parser.add_argument(
        "--unsafe-events",
        type=Path,
        default=Path(
            "/project/AV/drive-the-thoughts/data/benchmark/"
            "ood_hazard_2604_unsafe_traces.json"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("runs/reasoning_coverage/ood_hazard_2604.features.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/reasoning_coverage/ood_hazard_2604.program.json"),
    )
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--scene-repeats", type=int, default=200)
    parser.add_argument("--frame-repeats", type=int, default=50)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--bootstraps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_ood_evaluation(
        args.root,
        args.benchmark,
        args.unsafe_events,
        args.cache,
        bins=args.bins,
        scene_repeats=args.scene_repeats,
        frame_repeats=args.frame_repeats,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        seed=args.seed,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"Strict OOD scenes: {report['scene_level']['n_cases']}")
    print(f"Scene outcomes: {report['scene_level']['outcome_counts']}")
    print(
        "Strict OOD frames / known unsafe: "
        f"{report['frame_level_challenge']['n_cases']} / "
        f"{report['frame_level_challenge']['n_known_manual_unsafe_events']}"
    )
    print("Closed-loop failure APFD:")
    outcome = report["scene_level"]["outcomes"]["closed_loop_failure"]
    for arm, values in outcome["arms"].items():
        print(f"  {arm:26s} {values['apfd']['mean']:.3f}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
