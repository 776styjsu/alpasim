# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Evaluate a fixed causal-obligation denominator on Alpamayo OOD plans."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from alpasim_utils.geometry import quat_to_yaw
from alpasim_utils.logs import async_read_pb_log
from trajectory_safety.planned_center_labels import (
    COLLISION_OUTCOME,
    COMBINED_OUTCOME,
    OFFROAD_OUTCOME,
)

from .obligations import (
    ALPAMAYO_LATERAL_DECISIONS,
    ALPAMAYO_LONGITUDINAL_DECISIONS,
    CATALOG_VERSION,
    CRITICAL_COMPONENT_DOMAINS,
    REASONING_STATUSES,
    ActorState,
    ApplicableObligation,
    applicability_items,
    applicable_obligations,
    catalog,
    catalog_sha256,
    decision_items,
    decision_schema_sha256,
    denominator_items,
    reasoning_status,
    status_items,
)
from .ood_evaluation import (
    _development_scene_ids,
    _file_sha256,
    _load_or_extract_frames,
    build_frame_coverage,
    lazy_greedy_coverage_order,
)
from .planned_safety_evaluation import (
    COMBINED_10CM_OUTCOME,
    OFFROAD_10CM_OUTCOME,
    OUTCOME_FIELDS,
    _load_label_rows,
    _order_sha256,
    _outcome_result,
)

EXPOSURE_CACHE_KIND = "alpamayo_causal_obligation_exposures"
REPORT_KIND = "alpamayo_causal_obligation_evaluation"
EXPOSURE_CACHE_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
DEFAULT_ROOT = Path(
    "/project/AV/drive-the-thoughts/alpasim/tutorial_alpamayo_ood_hazard_2604"
)


def _frame_index(scene_path: Path) -> dict[int, tuple[str, str]]:
    result: dict[int, tuple[str, str]] = {}
    for path in sorted(scene_path.glob("extracted_frames/**/metadata.json")):
        with path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        timestamp = metadata["time_now_us"]
        frame_id = metadata["clip_id"]
        if not isinstance(timestamp, int) or not isinstance(frame_id, str):
            raise ValueError(f"Invalid frame identity in {path}")
        if timestamp in result:
            raise ValueError(f"Duplicate timestamp {timestamp} under {scene_path}")
        result[timestamp] = (frame_id, str(path))
    if not result:
        raise ValueError(f"No extracted frames under {scene_path}")
    return result


async def _extract_asl_exposures(
    asl_path: Path, frame_by_time: dict[int, tuple[str, str]]
) -> list[dict[str, Any]]:
    definitions: dict[str, tuple[str, float, float]] | None = None
    previous: dict[str, tuple[int, float, float]] = {}
    rows: list[dict[str, Any]] = []
    async for message in async_read_pb_log(str(asl_path)):
        message_type = message.WhichOneof("log_entry")
        if message_type == "rollout_metadata":
            definitions = {
                definition.actor_id: (
                    definition.actor_label,
                    float(definition.aabb.size_x),
                    float(definition.aabb.size_y),
                )
                for definition in message.rollout_metadata.actor_definitions.actor_aabb
            }
            continue
        if message_type != "actor_poses":
            continue
        timestamp = int(message.actor_poses.timestamp_us)
        current_positions = {
            actor.actor_id: (
                float(actor.actor_pose.vec.x),
                float(actor.actor_pose.vec.y),
            )
            for actor in message.actor_poses.actor_poses
        }
        if timestamp in frame_by_time:
            if definitions is None:
                raise ValueError(f"Actor poses precede metadata in {asl_path}")
            states: list[ActorState] = []
            for actor in message.actor_poses.actor_poses:
                actor_id = actor.actor_id
                if actor_id not in definitions:
                    raise ValueError(
                        f"Missing definition for actor {actor_id} in {asl_path}"
                    )
                actor_label, length_m, width_m = definitions[actor_id]
                x_m, y_m = current_positions[actor_id]
                old = previous.get(actor_id)
                if old is None:
                    velocity_x = 0.0
                    velocity_y = 0.0
                else:
                    old_timestamp, old_x, old_y = old
                    elapsed_s = (timestamp - old_timestamp) / 1e6
                    if not 0.0 < elapsed_s <= 0.2:
                        velocity_x = 0.0
                        velocity_y = 0.0
                    else:
                        velocity_x = (x_m - old_x) / elapsed_s
                        velocity_y = (y_m - old_y) / elapsed_s
                states.append(
                    ActorState(
                        actor_id=actor_id,
                        actor_label=actor_label,
                        x_m=x_m,
                        y_m=y_m,
                        yaw_rad=float(quat_to_yaw(actor.actor_pose.quat)),
                        velocity_x_mps=velocity_x,
                        velocity_y_mps=velocity_y,
                        length_m=length_m,
                        width_m=width_m,
                    )
                )
            frame_id, metadata_path = frame_by_time[timestamp]
            obligations = applicable_obligations(states)
            rows.append(
                {
                    "frame_id": frame_id,
                    "time_now_us": timestamp,
                    "metadata_path": metadata_path,
                    "obligations": [asdict(item) for item in obligations],
                }
            )
        previous = {
            actor_id: (timestamp, x_m, y_m)
            for actor_id, (x_m, y_m) in current_positions.items()
        }
    return rows


def _extract_scene(arguments: tuple[str, str]) -> dict[str, Any]:
    scene_path = Path(arguments[0])
    asl_path = Path(arguments[1])
    if not asl_path.is_file():
        raise ValueError(f"Missing selected rollout log: {asl_path}")
    if scene_path not in asl_path.parents:
        raise ValueError(f"Selected ASL {asl_path} is not under {scene_path}")
    frame_by_time = _frame_index(scene_path)
    rows = asyncio.run(_extract_asl_exposures(asl_path, frame_by_time))
    expected = {frame_id for frame_id, _ in frame_by_time.values()}
    actual = {row["frame_id"] for row in rows}
    if actual != expected:
        raise ValueError(
            f"Exposure extraction mismatch in {scene_path}: "
            f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
        )
    scene_ids = {frame_id.rsplit("_t", maxsplit=1)[0] for frame_id in actual}
    if len(scene_ids) != 1:
        raise ValueError(f"Multiple clip IDs under {scene_path}: {scene_ids}")
    return {
        "scene_id": scene_ids.pop(),
        "scene_path": str(scene_path),
        "asl_path": str(asl_path),
        "asl_size_bytes": asl_path.stat().st_size,
        "asl_mtime_ns": asl_path.stat().st_mtime_ns,
        "frames": rows,
    }


def _asl_selection_sha256(scene_inputs: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for scene in sorted(scene_inputs, key=lambda item: item["scene_id"]):
        for key in (
            "scene_id",
            "scene_path",
            "asl_path",
            "asl_size_bytes",
            "asl_mtime_ns",
        ):
            digest.update(str(scene[key]).encode())
            digest.update(b"\0")
    return digest.hexdigest()


def build_exposure_cache(
    root: Path, scene_inputs: list[dict[str, Any]], *, workers: int
) -> dict[str, Any]:
    """Extract independent actor-obligation triggers from every OOD ASL."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not scene_inputs:
        raise ValueError("No selected ASL inputs")
    arguments: list[tuple[str, str]] = []
    for scene in scene_inputs:
        scene_path = Path(scene["scene_path"])
        asl_path = Path(scene["asl_path"])
        if root not in scene_path.parents:
            raise ValueError(f"Selected scene {scene_path} is not under {root}")
        if asl_path.stat().st_size != scene["asl_size_bytes"]:
            raise ValueError(
                f"Selected ASL size changed since safety labeling: {asl_path}"
            )
        if asl_path.stat().st_mtime_ns != scene["asl_mtime_ns"]:
            raise ValueError(
                f"Selected ASL mtime changed since safety labeling: {asl_path}"
            )
        arguments.append((str(scene_path), str(asl_path)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        scenes = list(executor.map(_extract_scene, arguments))
    frames = [row for scene in scenes for row in scene["frames"]]
    frame_ids = [row["frame_id"] for row in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("Duplicate frame IDs across extracted exposure scenes")
    return {
        "schema_version": EXPOSURE_CACHE_SCHEMA_VERSION,
        "kind": EXPOSURE_CACHE_KIND,
        "catalog_version": CATALOG_VERSION,
        "catalog_sha256": catalog_sha256(),
        "inputs": {
            "root": str(root),
            "asl_selection_sha256": _asl_selection_sha256(scene_inputs),
            "extractor_sha256": _file_sha256(Path(__file__)),
        },
        "scenes": [
            {key: value for key, value in scene.items() if key != "frames"}
            for scene in scenes
        ],
        "frames": frames,
    }


def _load_or_build_exposures(
    root: Path,
    cache_path: Path,
    scene_inputs: list[dict[str, Any]],
    workers: int,
) -> tuple[dict[str, Any], dict[str, list[ApplicableObligation]]]:
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        expected = (
            payload.get("kind") == EXPOSURE_CACHE_KIND
            and payload.get("schema_version") == EXPOSURE_CACHE_SCHEMA_VERSION
            and payload.get("catalog_version") == CATALOG_VERSION
            and payload.get("catalog_sha256") == catalog_sha256()
            and payload.get("inputs", {}).get("root") == str(root)
            and payload.get("inputs", {}).get("asl_selection_sha256")
            == _asl_selection_sha256(scene_inputs)
        )
        if not expected:
            raise ValueError(
                f"Exposure cache does not match {CATALOG_VERSION} and root {root}: "
                f"{cache_path}"
            )
    else:
        payload = build_exposure_cache(root, scene_inputs, workers=workers)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
    by_frame: dict[str, list[ApplicableObligation]] = {}
    for row in payload["frames"]:
        frame_id = row["frame_id"]
        if frame_id in by_frame:
            raise ValueError(f"Duplicate exposure frame: {frame_id}")
        by_frame[frame_id] = [ApplicableObligation(**raw) for raw in row["obligations"]]
    return payload, by_frame


def _orders(
    case_ids: Iterable[str],
    coverage: dict[str, dict[str, frozenset[str]]],
    *,
    repeats: int,
    seed: int,
) -> dict[str, list[list[str]]]:
    ordered_ids = sorted(case_ids)
    result: dict[str, list[list[str]]] = {
        arm: [] for arm in ("random", *coverage.keys())
    }
    for repeat in range(repeats):
        repeat_seed = seed + repeat
        shuffled = list(ordered_ids)
        random.Random(repeat_seed).shuffle(shuffled)
        result["random"].append(shuffled)
        for arm, item_sets in coverage.items():
            result[arm].append(
                lazy_greedy_coverage_order(ordered_ids, item_sets, repeat_seed)
            )
    return result


def _denominator_report(
    coverage: dict[str, frozenset[str]], denominator: frozenset[str]
) -> dict[str, Any]:
    observed = frozenset().union(*coverage.values())
    outside = observed - denominator
    if outside:
        raise AssertionError(
            f"Observed items outside frozen denominator: {sorted(outside)}"
        )
    variants = {item for item in denominator if item.startswith("cro:variant=")}
    observed_variants = observed & variants
    status_cells = {
        item for item in denominator if item.startswith("cro-status:variant=")
    }
    observed_status_cells = observed & status_cells
    decision_cells = {
        item for item in denominator if item.startswith("cro-decision:variant=")
    }
    observed_decision_cells = observed & decision_cells
    return {
        "n_denominator_items": len(denominator),
        "n_observed_items": len(observed),
        "coverage_fraction": len(observed) / len(denominator),
        "n_exact_variants": len(variants),
        "n_observed_exact_variants": len(observed_variants),
        "exact_variant_coverage_fraction": len(observed_variants) / len(variants),
        "n_status_cells": len(status_cells),
        "n_observed_status_cells": len(observed_status_cells),
        "status_cell_coverage_fraction": (
            len(observed_status_cells) / len(status_cells) if status_cells else None
        ),
        "n_decision_cells": len(decision_cells),
        "n_observed_decision_cells": len(observed_decision_cells),
        "decision_cell_coverage_fraction": (
            len(observed_decision_cells) / len(decision_cells)
            if decision_cells
            else None
        ),
        "unobserved_exact_variants": sorted(variants - observed_variants),
    }


def _coverage_progress(
    order: list[str], item_sets: dict[str, frozenset[str]]
) -> dict[str, Any]:
    """Summarize coverage saturation for one representative tie seed."""
    observed_universe = frozenset().union(*item_sets.values())
    covered: set[str] = set()
    budgets = {
        "10_frames": min(10, len(order)),
        "100_frames": min(100, len(order)),
        "1_percent": max(1, round(0.01 * len(order))),
        "5_percent": max(1, round(0.05 * len(order))),
    }
    fractions: dict[str, float] = {}
    saturation_rank: int | None = None
    budgets_by_rank = {rank: name for name, rank in budgets.items()}
    for rank, frame_id in enumerate(order, start=1):
        covered.update(item_sets[frame_id])
        for name in (budgets_by_rank.get(rank),):
            if name is not None:
                fractions[name] = len(covered) / len(observed_universe)
        if covered == observed_universe:
            saturation_rank = rank
            break
    for name, rank in budgets.items():
        if name not in fractions:
            fractions[name] = (
                1.0
                if saturation_rank is not None and saturation_rank <= rank
                else len(covered) / len(observed_universe)
            )
    return {
        "denominator_scope": "items_observed_in_this_cohort",
        "n_observed_items": len(observed_universe),
        "saturation_rank": saturation_rank,
        "saturation_fraction_of_cases": saturation_rank / len(order),
        "coverage_fraction_at_budget": fractions,
    }


def run_obligation_evaluation(
    root: Path,
    benchmark_path: Path,
    safety_labels_path: Path,
    feature_cache_path: Path,
    exposure_cache_path: Path,
    *,
    bins: int,
    repeats: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    """Run the fixed-denominator bake-off on valid source-disjoint frames."""
    frames, feature_audit = _load_or_extract_frames(root, feature_cache_path, workers)
    development_scenes = _development_scene_ids(benchmark_path)
    label_payload, safety_labels = _load_label_rows(safety_labels_path)
    exposure_payload, exposures = _load_or_build_exposures(
        root,
        exposure_cache_path,
        label_payload["inputs"]["scenes"],
        workers,
    )
    source_disjoint = [
        frame for frame in frames if frame.scene_id not in development_scenes
    ]
    missing_labels = {
        frame.frame_id for frame in source_disjoint
    } - safety_labels.keys()
    missing_exposures = {frame.frame_id for frame in source_disjoint} - exposures.keys()
    if missing_labels:
        raise ValueError(f"Missing safety labels: {sorted(missing_labels)[:5]}")
    if missing_exposures:
        raise ValueError(
            f"Missing obligation exposures: {sorted(missing_exposures)[:5]}"
        )
    valid_frames = [
        frame for frame in source_disjoint if safety_labels[frame.frame_id]["valid"]
    ]
    if not valid_frames:
        raise ValueError("No valid source-disjoint frames")
    baseline, discretization = build_frame_coverage(valid_frames, bins)
    applicability_coverage = {
        frame.frame_id: applicability_items(exposures[frame.frame_id])
        for frame in valid_frames
    }
    status_coverage = {
        frame.frame_id: status_items(frame.program, exposures[frame.frame_id])
        for frame in valid_frames
    }
    decision_coverage = {
        frame.frame_id: decision_items(frame.program, exposures[frame.frame_id])
        for frame in valid_frames
    }
    coverage = {
        "trajectory": baseline["trajectory"],
        "text_bigram": baseline["text_bigram"],
        "program_path": baseline["program_path"],
        "cro_applicability": applicability_coverage,
        "cro_status": status_coverage,
        "cro_decision": decision_coverage,
        "cro_status_path_hybrid": {
            frame.frame_id: status_coverage[frame.frame_id]
            | baseline["program_path"][frame.frame_id]
            for frame in valid_frames
        },
    }
    case_ids = [frame.frame_id for frame in valid_frames]
    orders = _orders(case_ids, coverage, repeats=repeats, seed=seed)

    scene_by_frame = {frame.frame_id: frame.scene_id for frame in valid_frames}
    rows = []
    for frame_id in case_ids:
        row = dict(safety_labels[frame_id])
        row["scene_id"] = scene_by_frame[frame_id]
        rows.append(row)
    failures_by_outcome = {
        outcome: {row["frame_id"] for row in rows if bool(row[field])}
        for outcome, field in OUTCOME_FIELDS.items()
    }
    failures_by_outcome[OFFROAD_10CM_OUTCOME] = {
        row["frame_id"]
        for row in rows
        if row["max_center_distance_outside_drivable_area_m"] > 0.1
    }
    failures_by_outcome[COMBINED_10CM_OUTCOME] = (
        failures_by_outcome[COLLISION_OUTCOME]
        | failures_by_outcome[OFFROAD_10CM_OUTCOME]
    )
    outcomes = {
        outcome: _outcome_result(orders, failures)
        for outcome, failures in failures_by_outcome.items()
        if failures
    }

    frame_by_id = {frame.frame_id: frame for frame in valid_frames}
    status_counts: Counter[str] = Counter()
    status_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    family_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    for frame_id, applicable in exposures.items():
        if frame_id not in frame_by_id:
            continue
        for obligation in applicable:
            family_counts[obligation.family] += 1
            variant_counts[obligation.obligation_id] += 1
            status_counts[
                reasoning_status(frame_by_id[frame_id].program, obligation)
            ] += 1
            status_by_family[obligation.family][
                reasoning_status(frame_by_id[frame_id].program, obligation)
            ] += 1
    triggered_frames = sum(bool(exposures[frame_id]) for frame_id in case_ids)
    application_denominator = denominator_items(include_status=False)
    status_denominator = denominator_items(include_status=True)
    decision_denominator = denominator_items(
        include_status=True, include_decisions=True
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "catalog": {
            "version": CATALOG_VERSION,
            "sha256": catalog_sha256(),
            "n_exact_variants": len(catalog()),
            "definitions": [asdict(item) for item in catalog()],
            "reasoning_statuses": list(REASONING_STATUSES),
            "decision_schema_sha256": decision_schema_sha256(),
            "longitudinal_decisions": list(ALPAMAYO_LONGITUDINAL_DECISIONS),
            "lateral_decisions": list(ALPAMAYO_LATERAL_DECISIONS),
            "n_nonempty_decision_pairs": (
                len(ALPAMAYO_LONGITUDINAL_DECISIONS) * len(ALPAMAYO_LATERAL_DECISIONS)
                - 1
            ),
            "critical_component_domains": [
                {"domain": domain, "observability": observability}
                for domain, observability in CRITICAL_COMPONENT_DOMAINS
            ],
            "applicability_semantics": (
                "nearest aligned vehicle in an AABB-width ego corridor, or a "
                "transverse moving actor with constant-velocity closest approach "
                "within 5 s and the conservative AABB collision radius"
            ),
        },
        "inputs": {
            "root": str(root),
            "development_benchmark": str(benchmark_path),
            "development_benchmark_sha256": _file_sha256(benchmark_path),
            "planned_safety_labels": str(safety_labels_path),
            "planned_safety_labels_sha256": _file_sha256(safety_labels_path),
            "labeler_sha256_recorded_in_labels": label_payload["inputs"][
                "labeler_sha256"
            ],
            "feature_cache": str(feature_cache_path),
            "feature_cache_fingerprint": feature_audit["fingerprint"],
            "exposure_cache": str(exposure_cache_path),
            "exposure_extractor_sha256": exposure_payload["inputs"]["extractor_sha256"],
            "evaluator_sha256": _file_sha256(Path(__file__)),
        },
        "config": {
            "bins": bins,
            "repeats": repeats,
            "seed": seed,
            "workers": workers,
            "primary_arms": [
                "cro_applicability",
                "cro_status",
                "cro_decision",
            ],
            "sensitivity_arm": "cro_status_path_hybrid",
        },
        "dataset_audit": {
            "n_all_frames": len(frames),
            "n_source_disjoint_frames": len(source_disjoint),
            "n_valid_source_disjoint_frames": len(valid_frames),
            "n_valid_source_disjoint_scenes": len(
                {frame.scene_id for frame in valid_frames}
            ),
            "n_frames_with_applicable_obligation": triggered_frames,
            "fraction_frames_with_applicable_obligation": triggered_frames
            / len(valid_frames),
            "n_applicable_obligations": sum(family_counts.values()),
            "family_counts": dict(sorted(family_counts.items())),
            "reasoning_status_counts": dict(sorted(status_counts.items())),
            "reasoning_status_counts_by_family": {
                family: dict(sorted(counts.items()))
                for family, counts in sorted(status_by_family.items())
            },
            "exact_variant_counts": dict(sorted(variant_counts.items())),
        },
        "denominator_coverage": {
            "cro_applicability": _denominator_report(
                applicability_coverage, application_denominator
            ),
            "cro_status": _denominator_report(status_coverage, status_denominator),
            "cro_decision": _denominator_report(
                decision_coverage, decision_denominator
            ),
        },
        "discretization": discretization,
        "coverage_dynamics": {
            arm: _coverage_progress(orders[arm][0], item_sets)
            for arm, item_sets in coverage.items()
        },
        "outcomes": outcomes,
        "representative_orders": {
            arm: {
                "sha256": _order_sha256(arm_orders[0]),
                "first_100_frame_ids": arm_orders[0][:100],
            }
            for arm, arm_orders in orders.items()
        },
        "limitations": [
            "Version 1 independently observes only critical-object obligations; "
            "six catalog domains remain uninstrumented.",
            "Applicability is a deterministic geometry-and-kinematics proxy, not a human driving judgment.",
            "The constant-velocity crossing test does not model actor reactions or map right-of-way.",
            "Expected-decision status evaluates textual grounding, not whether the emitted trajectory is safe.",
            "The decision-conditioned denominator includes all 71 published non-empty pairs, "
            "including pairs that may be infeasible for a particular obligation.",
            "Outcomes describe emitted discrete plans against "
            "recorded non-reactive agents and center off-road geometry.",
            "This suite and its outcomes were inspected while designing the abstraction; results are exploratory.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen causal-obligation denominator"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark.json"))
    parser.add_argument(
        "--safety-labels",
        type=Path,
        default=Path(
            "runs/reasoning_coverage/ood_hazard_2604.planned_safety_3s_100ms.json"
        ),
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("runs/reasoning_coverage/ood_hazard_2604.features.json"),
    )
    parser.add_argument(
        "--exposure-cache",
        type=Path,
        default=Path("runs/reasoning_coverage/ood_hazard_2604.cro_v1_exposures.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/reasoning_coverage/ood_hazard_2604.cro_v1.json"),
    )
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_obligation_evaluation(
        args.root,
        args.benchmark,
        args.safety_labels,
        args.feature_cache,
        args.exposure_cache,
        bins=args.bins,
        repeats=args.repeats,
        seed=args.seed,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report["dataset_audit"], indent=2))
    for outcome in (COLLISION_OUTCOME, OFFROAD_OUTCOME, COMBINED_OUTCOME):
        result = report["outcomes"][outcome]
        print(f"{outcome}: {result['n_positive_frames']} positive frames")
        for arm in (
            "random",
            "trajectory",
            "text_bigram",
            "program_path",
            "cro_applicability",
            "cro_status",
            "cro_decision",
            "cro_status_path_hybrid",
        ):
            value = result["arms"][arm]["frame_apfd"]["mean"]
            print(f"  {arm:28s} {value:.3f}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
