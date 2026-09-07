# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Measure temporal leakage in scene-level reasoning coverage.

The primary OOD evaluator unions coverage over an entire rollout.  That is a
useful retrospective characterization, but it can include reasoning emitted
after the event used as the outcome.  This module repeats scene prioritization
with an equal early prefix and audits metric-event timestamps against CoT frame
timestamps.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rollout_monitoring.analysis import OutcomeSeries, load_outcome_series

from .ood_evaluation import (
    OOD_ARMS,
    OUTCOME_NAMES,
    SceneRecord,
    _coverage_metadata,
    _file_collection_sha256,
    _file_sha256,
    _manual_unsafe_ids,
    _orders,
    aggregate_scene_coverage,
    build_frame_coverage,
    evaluate_orders,
    load_ood_cases,
)

OUTCOME_LABEL_SCOPE = {
    "closed_loop_failure": "exhaustive",
    "manual_unsafe_scene": "positive_unlabeled",
    "closed_loop_safety_event": "exhaustive",
}
SAFETY_EVENT_COMPONENTS = (
    "collision_front",
    "collision_lateral",
    "offroad",
    "wrong_lane",
    "safety_monitor_triggered",
)


@dataclass(frozen=True)
class TemporalScene:
    scene: SceneRecord
    frame_times_us: tuple[int, ...]
    failure_event_us: int | None
    safety_event_us: int | None
    manual_unsafe_event_us: int | None

    def event_time(self, outcome: str) -> int | None:
        if outcome == "closed_loop_failure":
            return self.failure_event_us
        if outcome == "closed_loop_safety_event":
            return self.safety_event_us
        if outcome == "manual_unsafe_scene":
            return self.manual_unsafe_event_us
        raise ValueError(f"Unknown outcome: {outcome}")


def prefix_frame_count(n_frames: int, fraction: float) -> int:
    """Return the equal-prefix size, rounding a non-empty prefix upward."""
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return math.ceil(n_frames * fraction)


def common_strict_pre_event_frames(
    frame_times_by_scene: dict[str, tuple[int, ...]],
    event_times_by_scene: dict[str, int],
) -> int:
    """Largest equal prefix strictly before every supplied positive event.

    The same returned number must be used for positive and negative scenes.
    This avoids giving no-event scenes a longer trace than event scenes.
    """
    if not event_times_by_scene:
        raise ValueError("at least one event time is required")
    missing = event_times_by_scene.keys() - frame_times_by_scene.keys()
    if missing:
        raise ValueError(f"missing frame times for scenes: {sorted(missing)}")
    return min(
        sum(timestamp < event_times_by_scene[scene_id] for timestamp in times)
        for scene_id, times in frame_times_by_scene.items()
        if scene_id in event_times_by_scene
    )


def _metric_paths(scene: SceneRecord) -> list[Path]:
    paths = list(scene.scene_path.glob("rollouts/*/*/metrics.parquet"))
    if len(paths) != 1:
        raise ValueError(
            f"Expected one metrics.parquet for {scene.scene_id}, found {len(paths)}"
        )
    return paths


def _first_component_event(
    series: OutcomeSeries, components: Iterable[str]
) -> int | None:
    event_times: list[int] = []
    for component in components:
        values = series.raw.get(component)
        if values is None:
            continue
        timestamps, metric_values = values
        mask = (
            (metric_values > 0.0)
            & (timestamps >= series.eval_start_us)
            & (timestamps <= series.window_end_us)
        )
        if mask.any():
            event_times.append(int(timestamps[mask].min()))
    return min(event_times) if event_times else None


def _frame_times(scene: SceneRecord) -> tuple[int, ...]:
    times: list[int] = []
    for frame in scene.frames:
        with frame.metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        timestamp = metadata.get("time_now_us")
        if not isinstance(timestamp, int):
            raise ValueError(f"Missing integer time_now_us: {frame.metadata_path}")
        times.append(timestamp)
    if times != sorted(times) or len(set(times)) != len(times):
        raise ValueError(f"Non-monotonic frame timestamps for {scene.scene_id}")
    return tuple(times)


def _temporal_scenes(
    scenes: list[SceneRecord], unsafe_path: Path
) -> list[TemporalScene]:
    unsafe_frame_ids, _ = _manual_unsafe_ids(unsafe_path)
    result: list[TemporalScene] = []
    for scene in scenes:
        frame_times = _frame_times(scene)
        time_by_frame = {
            frame.frame_id: timestamp
            for frame, timestamp in zip(scene.frames, frame_times, strict=True)
        }
        manual_times = [
            timestamp
            for frame_id, timestamp in time_by_frame.items()
            if frame_id in unsafe_frame_ids
        ]

        series: OutcomeSeries | None = None
        if scene.outcome("closed_loop_failure") or scene.outcome(
            "closed_loop_safety_event"
        ):
            series = load_outcome_series(_metric_paths(scene)[0])
        failure_event = (
            series.failure_time_us("offroad_or_collision_at_fault")
            if series is not None and scene.outcome("closed_loop_failure")
            else None
        )
        safety_event = (
            _first_component_event(series, SAFETY_EVENT_COMPONENTS)
            if series is not None and scene.outcome("closed_loop_safety_event")
            else None
        )
        result.append(
            TemporalScene(
                scene=scene,
                frame_times_us=frame_times,
                failure_event_us=failure_event,
                safety_event_us=safety_event,
                manual_unsafe_event_us=min(manual_times) if manual_times else None,
            )
        )
    return result


def _event_audit(
    temporal_scenes: list[TemporalScene], prefix_frames: int
) -> dict[str, Any]:
    frame_times_by_scene = {
        item.scene.scene_id: item.frame_times_us for item in temporal_scenes
    }
    audit: dict[str, Any] = {}
    for outcome in OUTCOME_NAMES:
        positives = [item for item in temporal_scenes if item.scene.outcome(outcome)]
        known = {
            item.scene.scene_id: event
            for item in positives
            if (event := item.event_time(outcome)) is not None
        }
        details: list[dict[str, Any]] = []
        for item in positives:
            event = item.event_time(outcome)
            if event is None:
                details.append({"scene_id": item.scene.scene_id, "event_time_us": None})
                continue
            prefix_times = item.frame_times_us[:prefix_frames]
            details.append(
                {
                    "scene_id": item.scene.scene_id,
                    "event_time_us": event,
                    "n_frames_strictly_before_event": sum(
                        timestamp < event for timestamp in item.frame_times_us
                    ),
                    "n_frames_strictly_after_event": sum(
                        timestamp > event for timestamp in item.frame_times_us
                    ),
                    "event_at_or_before_prefix_end": event <= prefix_times[-1],
                    "prefix_contains_post_event_frame": any(
                        timestamp > event for timestamp in prefix_times
                    ),
                }
            )
        known_details = [
            detail for detail in details if detail["event_time_us"] is not None
        ]
        before = [
            int(detail["n_frames_strictly_before_event"]) for detail in known_details
        ]
        common_horizon = (
            common_strict_pre_event_frames(frame_times_by_scene, known)
            if len(known) == len(positives)
            else None
        )
        audit[outcome] = {
            "n_positive_scenes": len(positives),
            "n_with_known_event_time": len(known),
            "n_with_full_trace_post_event_frames": sum(
                int(detail["n_frames_strictly_after_event"]) > 0
                for detail in known_details
            ),
            "n_event_at_or_before_prefix_end": sum(
                bool(detail["event_at_or_before_prefix_end"])
                for detail in known_details
            ),
            "n_prefix_contains_post_event_frames": sum(
                bool(detail["prefix_contains_post_event_frame"])
                for detail in known_details
            ),
            "pre_event_frame_count": {
                "minimum": min(before) if before else None,
                "median": float(np.median(before)) if before else None,
                "maximum": max(before) if before else None,
            },
            "common_strict_pre_event_frames": common_horizon,
            "positive_scene_details": details,
        }
    return audit


def _truncate_scenes(scenes: list[SceneRecord], n_frames: int) -> list[SceneRecord]:
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    if any(len(scene.frames) < n_frames for scene in scenes):
        raise ValueError("prefix exceeds at least one scene trace")
    return [
        SceneRecord(
            scene_id=scene.scene_id,
            scene_path=scene.scene_path,
            frames=scene.frames[:n_frames],
            outcomes=scene.outcomes,
        )
        for scene in scenes
    ]


def _tied_random_orders(
    scene_ids: list[str], repeats: int, seed: int
) -> dict[str, list[list[str]]]:
    random_orders: list[list[str]] = []
    for repeat in range(repeats):
        order = sorted(scene_ids)
        random.Random(seed + repeat).shuffle(order)
        random_orders.append(order)
    return {arm: [list(order) for order in random_orders] for arm in OOD_ARMS}


def _prefix_orders(
    scenes: list[SceneRecord],
    n_frames: int,
    *,
    bins: int,
    repeats: int,
    seed: int,
) -> tuple[
    dict[str, list[list[str]]],
    dict[str, Any],
    dict[str, Any] | None,
]:
    scene_ids = [scene.scene_id for scene in scenes]
    if n_frames == 0:
        return (
            _tied_random_orders(scene_ids, repeats, seed),
            {
                arm: {"n_items": 0, "mean_items_per_case": 0.0}
                for arm in OOD_ARMS
                if arm != "random"
            },
            None,
        )
    truncated = _truncate_scenes(scenes, n_frames)
    frames = [frame for scene in truncated for frame in scene.frames]
    frame_coverage, discretization = build_frame_coverage(frames, bins)
    scene_coverage = aggregate_scene_coverage(truncated, frame_coverage)
    return (
        _orders(scene_ids, scene_coverage, repeats, seed),
        _coverage_metadata(scene_coverage),
        discretization,
    )


def _failures(scenes: list[SceneRecord]) -> dict[str, set[str]]:
    return {
        outcome: {scene.scene_id for scene in scenes if scene.outcome(outcome)}
        for outcome in OUTCOME_NAMES
    }


def _evaluate(
    orders: dict[str, list[list[str]]],
    failures: dict[str, set[str]],
    scene_ids: list[str],
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    return evaluate_orders(
        orders,
        failures,
        {outcome: OUTCOME_LABEL_SCOPE[outcome] for outcome in failures},
        {scene_id: scene_id for scene_id in scene_ids},
        permutations,
        bootstraps,
        seed,
    )


def run_temporal_sensitivity(
    root: Path,
    benchmark_path: Path,
    unsafe_path: Path,
    cache_path: Path,
    *,
    prefix_fraction: float,
    bins: int,
    repeats: int,
    permutations: int,
    bootstraps: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    _, scenes, dataset_audit = load_ood_cases(
        root, benchmark_path, unsafe_path, cache_path, workers
    )
    if not scenes:
        raise ValueError("No strict OOD scenes with outcomes")
    scene_lengths = {len(scene.frames) for scene in scenes}
    if len(scene_lengths) != 1:
        raise ValueError(
            "Equal-fraction sensitivity requires equal-length scene traces"
        )
    frames_per_scene = next(iter(scene_lengths))
    prefix_frames = prefix_frame_count(frames_per_scene, prefix_fraction)
    temporal_scenes = _temporal_scenes(scenes, unsafe_path)
    event_audit = _event_audit(temporal_scenes, prefix_frames)

    scene_ids = [scene.scene_id for scene in scenes]
    failures = _failures(scenes)
    prefix_orders, prefix_coverage, prefix_discretization = _prefix_orders(
        scenes,
        prefix_frames,
        bins=bins,
        repeats=repeats,
        seed=seed,
    )
    prefix_results = _evaluate(
        prefix_orders,
        failures,
        scene_ids,
        permutations=permutations,
        bootstraps=bootstraps,
        seed=seed,
    )

    common_results: dict[str, Any] = {}
    order_cache: dict[int, tuple[dict[str, list[list[str]]], dict[str, Any]]] = {}
    for outcome in OUTCOME_NAMES:
        horizon = event_audit[outcome]["common_strict_pre_event_frames"]
        if horizon is None:
            common_results[outcome] = {
                "n_frames_per_scene": None,
                "reason": "At least one positive scene has no timestamped event.",
            }
            continue
        if horizon not in order_cache:
            orders, coverage, _ = _prefix_orders(
                scenes,
                horizon,
                bins=bins,
                repeats=repeats,
                seed=seed,
            )
            order_cache[horizon] = orders, coverage
        orders, coverage = order_cache[horizon]
        outcome_results = _evaluate(
            orders,
            {outcome: failures[outcome]},
            scene_ids,
            permutations=permutations,
            bootstraps=bootstraps,
            seed=seed,
        )
        common_results[outcome] = {
            "n_frames_per_scene": horizon,
            "coverage": coverage,
            "result": outcome_results[outcome],
        }

    metric_paths = [path for scene in scenes for path in _metric_paths(scene)]
    summary_paths = [
        scene.scene_path / "aggregate" / "results-summary.json" for scene in scenes
    ]
    return {
        "schema_version": 1,
        "kind": "reasoning_program_ood_temporal_sensitivity",
        "inputs": {
            "root": str(root),
            "development_benchmark": str(benchmark_path),
            "development_benchmark_sha256": _file_sha256(benchmark_path),
            "manual_unsafe_events": str(unsafe_path),
            "manual_unsafe_events_sha256": _file_sha256(unsafe_path),
            "ood_feature_dependency_fingerprint": dataset_audit["fingerprint"],
            "outcome_summaries_sha256": _file_collection_sha256(summary_paths, root),
            "metrics_parquet_sha256": _file_collection_sha256(metric_paths, root),
            "program_compiler_sha256": _file_sha256(
                Path(__file__).with_name("program.py")
            ),
            "feature_extractor_sha256": _file_sha256(
                Path(__file__).with_name("analysis.py")
            ),
            "ood_evaluator_sha256": _file_sha256(
                Path(__file__).with_name("ood_evaluation.py")
            ),
            "statistics_sha256": _file_sha256(
                Path(__file__).with_name("statistics.py")
            ),
            "event_window_loader_sha256": _file_sha256(
                Path(__file__).parents[1] / "rollout_monitoring" / "analysis.py"
            ),
            "temporal_evaluator_sha256": _file_sha256(Path(__file__)),
        },
        "config": {
            "prefix_fraction": prefix_fraction,
            "bins": bins,
            "repeats": repeats,
            "permutations": permutations,
            "bootstraps": bootstraps,
            "seed": seed,
            "workers": workers,
        },
        "cohort": {
            "n_scenes": len(scenes),
            "frames_per_scene": frames_per_scene,
            "outcome_counts": {
                outcome: len(case_ids) for outcome, case_ids in failures.items()
            },
            "strict_scene_ids": sorted(scene_ids),
            "development_overlap_excluded": True,
        },
        "protocol": {
            "prefix25": (
                f"Use the first ceil({prefix_fraction} * {frames_per_scene}) = "
                f"{prefix_frames} CoT frames for every scene. Refit quantile "
                "bins on only those retained prefix frames without outcome access."
            ),
            "common_strict_pre_event": (
                "For each outcome, use the minimum number of CoT frames strictly "
                "before the first event across all positive scenes, then give that "
                "same number of leading frames to every positive and negative scene."
            ),
            "event_time_alignment": (
                "Join metadata.time_now_us to raw metrics.parquet.timestamps_us; "
                "apply the evaluator's valid window before locating metric events."
            ),
            "tie_breaking": f"{repeats} aligned seeds from {seed} through "
            f"{seed + repeats - 1}.",
        },
        "dataset_audit": dataset_audit,
        "post_event_audit": event_audit,
        "prefix25": {
            "n_frames_per_scene": prefix_frames,
            "coverage": prefix_coverage,
            "discretization": prefix_discretization,
            "outcomes": prefix_results,
            "representative_orders": {
                arm: orders[0] for arm, orders in prefix_orders.items()
            },
        },
        "common_strict_pre_event": {
            "results_by_outcome": common_results,
            "warning": (
                "A zero-frame common horizon is a valid but uninformative result: "
                "all arms have empty coverage and share the random tie order."
            ),
        },
        "limitations": [
            "This is an adaptive exploratory sensitivity analysis, not a locked holdout.",
            "The 25% prefix is equal across scenes but is not guaranteed pre-event.",
            "A common strictly-pre-event horizon can collapse to zero under left-censoring.",
            "Manual unsafe labels are positive-unlabeled; unflagged scenes are not proven safe.",
            "Quantile bins are fit transductively on the retained OOD prefix.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit temporal leakage in OOD reasoning coverage"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/project/AV/drive-the-thoughts/alpasim/"
            "tutorial_alpamayo_ood_hazard_2604"
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
        default=Path("runs/reasoning_coverage/ood_hazard_2604.prefix25.json"),
    )
    parser.add_argument("--prefix-fraction", type=float, default=0.25)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--bootstraps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_temporal_sensitivity(
        args.root,
        args.benchmark,
        args.unsafe_events,
        args.cache,
        prefix_fraction=args.prefix_fraction,
        bins=args.bins,
        repeats=args.repeats,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        seed=args.seed,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print(
        f"Prefix: {report['prefix25']['n_frames_per_scene']} / "
        f"{report['cohort']['frames_per_scene']} frames per scene"
    )
    for outcome, values in report["prefix25"]["outcomes"].items():
        print(f"{outcome} ({values['n_failures']} positives):")
        for arm in (
            "random",
            "trajectory",
            "text_bigram",
            "program_edge",
            "program_path",
            "program_trace",
        ):
            mean = values["arms"][arm]["apfd"]["mean"]
            print(f"  {arm:18s} {mean:.3f}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
