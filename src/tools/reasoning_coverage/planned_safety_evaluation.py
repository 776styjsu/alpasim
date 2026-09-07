# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Evaluate reasoning coverage against exhaustive per-plan safety proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np
from trajectory_safety.planned_center_labels import (
    COLLISION_OUTCOME,
    COMBINED_OUTCOME,
    LABEL_KIND,
    OFFROAD_OUTCOME,
)

from .analysis import apfd
from .ood_evaluation import (
    _coverage_metadata,
    _development_scene_ids,
    _file_sha256,
    _load_or_extract_frames,
    build_frame_coverage,
    lazy_greedy_coverage_order,
)
from .table_program import (
    action_items,
    compile_table_program,
    load_action_runs,
    primary_reasoning_items,
    reasoning_action_items,
    sequence_reasoning_items,
    z1_category_x_z3_pair_items,
    z3_pair_items,
)

OFFROAD_10CM_OUTCOME = "planned_ego_center_more_than_10cm_outside_drivable_area"
COMBINED_10CM_OUTCOME = "open_loop_collision_or_center_more_than_10cm_offroad"
OUTCOME_FIELDS = {
    COLLISION_OUTCOME: "open_loop_collision_with_recorded_agents",
    OFFROAD_OUTCOME: "planned_center_offroad",
    COMBINED_OUTCOME: "open_loop_safety_violation",
}
TABLE_ARMS = (
    "table_z3_primary",
    "table_z3_sequence",
    "table_z1_x_z3_primary",
    "table_z1_x_z3_sequence",
    "table_primary",
    "table_sequence",
    "table_action",
    "table_reasoning_action",
)


def _add_table_coverage(
    frames: list[Any], coverage: dict[str, dict[str, frozenset[str]]]
) -> None:
    programs = {
        text: compile_table_program(text)
        for text in sorted({frame.program.text for frame in frames})
    }
    coverage.update({arm: {} for arm in TABLE_ARMS})
    for frame in frames:
        program = programs[frame.program.text]
        action_runs = load_action_runs(frame.metadata_path)
        values = {
            "table_z3_primary": z3_pair_items(program),
            "table_z3_sequence": z3_pair_items(program, include_continuations=True),
            "table_z1_x_z3_primary": z1_category_x_z3_pair_items(program),
            "table_z1_x_z3_sequence": z1_category_x_z3_pair_items(
                program, include_continuations=True
            ),
            "table_primary": primary_reasoning_items(program),
            "table_sequence": sequence_reasoning_items(program),
            "table_action": action_items(action_runs),
            "table_reasoning_action": reasoning_action_items(program, action_runs),
        }
        for arm, items in values.items():
            coverage[arm][frame.frame_id] = frozenset(items)


def _orders_for_coverage(
    case_ids: list[str],
    coverage: dict[str, dict[str, frozenset[str]]],
    repeats: int,
    seed: int,
) -> dict[str, list[list[str]]]:
    arms = ("random", *coverage)
    result: dict[str, list[list[str]]] = {arm: [] for arm in arms}
    ordered_ids = sorted(case_ids)
    for repeat in range(repeats):
        repeat_seed = seed + repeat
        random_order = list(ordered_ids)
        random.Random(repeat_seed).shuffle(random_order)
        result["random"].append(random_order)
        for arm, item_sets in coverage.items():
            result[arm].append(
                lazy_greedy_coverage_order(ordered_ids, item_sets, repeat_seed)
            )
    return result


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(fmean(values)),
        "std": float(pstdev(values)),
        "p2_5": float(np.quantile(array, 0.025)),
        "p97_5": float(np.quantile(array, 0.975)),
    }


def _recall(order: list[str], failures: set[str], budget: int) -> float:
    return len(set(order[:budget]) & failures) / len(failures)


def _outcome_result(
    orders: dict[str, list[list[str]]],
    failures: set[str],
) -> dict[str, Any]:
    n_cases = len(next(iter(orders.values()))[0])
    arm_results: dict[str, Any] = {}
    frame_apfd_by_arm: dict[str, list[float]] = {}
    for arm, arm_orders in orders.items():
        frame_values = [float(apfd(order, failures)) for order in arm_orders]
        frame_apfd_by_arm[arm] = frame_values
        arm_results[arm] = {
            "frame_apfd": _distribution(frame_values),
            "mean_frame_recall_at_budget": {
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
                for fraction in (0.01, 0.05, 0.1, 0.25)
            },
        }

    comparisons: dict[str, Any] = {}
    for baseline in ("random", "trajectory", "trajectory_grid", "text_bigram"):
        comparisons[baseline] = {}
        for arm in orders:
            if arm == baseline:
                continue
            frame_deltas = [
                candidate - reference
                for candidate, reference in zip(
                    frame_apfd_by_arm[arm],
                    frame_apfd_by_arm[baseline],
                    strict=True,
                )
            ]
            comparisons[baseline][arm] = {
                "frame_apfd_delta": _distribution(frame_deltas),
            }
    return {
        "n_positive_frames": len(failures),
        "label_scope": "exhaustive_within_valid_proxy_cohort",
        "arms": arm_results,
        "comparisons": comparisons,
    }


def _load_label_rows(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload["kind"] != LABEL_KIND or payload["schema_version"] != 1:
        raise ValueError(f"Unexpected planned-safety label schema: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for raw in payload["frames"]:
        frame_id = raw["frame_id"]
        if frame_id in rows:
            raise ValueError(f"Duplicate frame label: {frame_id}")
        rows[frame_id] = raw
    return payload, rows


def _order_sha256(order: list[str]) -> str:
    digest = hashlib.sha256()
    for frame_id in order:
        digest.update(frame_id.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def run_planned_safety_evaluation(
    root: Path,
    benchmark_path: Path,
    labels_path: Path,
    cache_path: Path,
    *,
    bins: int,
    repeats: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    """Evaluate outcome-blind coverage orders on valid source-disjoint plans."""
    frames, cache_audit = _load_or_extract_frames(root, cache_path, workers)
    development_scenes = _development_scene_ids(benchmark_path)
    strict_frames = [
        frame for frame in frames if frame.scene_id not in development_scenes
    ]
    label_payload, labels = _load_label_rows(labels_path)
    missing = {frame.frame_id for frame in strict_frames} - labels.keys()
    if missing:
        raise ValueError(f"Missing planned-safety labels: {sorted(missing)[:5]}")
    valid_frames = [frame for frame in strict_frames if labels[frame.frame_id]["valid"]]
    if not valid_frames:
        raise ValueError("No valid source-disjoint planned-safety cases")
    coverage, discretization = build_frame_coverage(valid_frames, bins)
    _add_table_coverage(valid_frames, coverage)
    case_ids = [frame.frame_id for frame in valid_frames]
    orders = _orders_for_coverage(case_ids, coverage, repeats, seed)
    valid_rows = []
    scene_id_by_frame = {frame.frame_id: frame.scene_id for frame in valid_frames}
    for frame_id in case_ids:
        row = dict(labels[frame_id])
        row["scene_id"] = scene_id_by_frame[frame_id]
        valid_rows.append(row)

    failures_by_outcome = {
        outcome: {row["frame_id"] for row in valid_rows if bool(row[field_name])}
        for outcome, field_name in OUTCOME_FIELDS.items()
    }
    failures_by_outcome[OFFROAD_10CM_OUTCOME] = {
        row["frame_id"]
        for row in valid_rows
        if row["max_center_distance_outside_drivable_area_m"] > 0.1
    }
    failures_by_outcome[COMBINED_10CM_OUTCOME] = (
        failures_by_outcome[COLLISION_OUTCOME]
        | failures_by_outcome[OFFROAD_10CM_OUTCOME]
    )

    outcomes: dict[str, Any] = {}
    for outcome, failures in failures_by_outcome.items():
        if not failures:
            continue
        outcomes[outcome] = _outcome_result(orders, failures)

    invalid_strict = [
        labels[frame.frame_id]
        for frame in strict_frames
        if not labels[frame.frame_id]["valid"]
    ]
    invalid_reasons: dict[str, int] = defaultdict(int)
    for row in invalid_strict:
        invalid_reasons[row["invalid_reason"]] += 1
    representative_orders = {
        arm: {
            "sha256": _order_sha256(arm_orders[0]),
            "first_100_frame_ids": arm_orders[0][:100],
        }
        for arm, arm_orders in orders.items()
    }
    return {
        "schema_version": 1,
        "kind": "reasoning_coverage_planned_safety_evaluation",
        "inputs": {
            "root": str(root),
            "development_benchmark": str(benchmark_path),
            "development_benchmark_sha256": _file_sha256(benchmark_path),
            "planned_safety_labels": str(labels_path),
            "planned_safety_labels_sha256": _file_sha256(labels_path),
            "labeler_sha256_recorded_in_labels": label_payload["inputs"][
                "labeler_sha256"
            ],
            "feature_cache": str(cache_path),
            "feature_cache_fingerprint": cache_audit["fingerprint"],
            "program_compiler_sha256": _file_sha256(
                Path(__file__).with_name("program.py")
            ),
            "table_program_compiler_sha256": _file_sha256(
                Path(__file__).with_name("table_program.py")
            ),
            "trajectory_grid_compiler_sha256": _file_sha256(
                Path(__file__).with_name("trajectory_grid.py")
            ),
            "evaluator_sha256": _file_sha256(Path(__file__)),
        },
        "config": {
            "bins": bins,
            "repeats": repeats,
            "seed": seed,
            "workers": workers,
            "primary_outcomes": [COLLISION_OUTCOME, OFFROAD_OUTCOME, COMBINED_OUTCOME],
            "sensitivity_outcomes": [OFFROAD_10CM_OUTCOME, COMBINED_10CM_OUTCOME],
        },
        "dataset_audit": {
            "n_all_extracted_frames": len(frames),
            "n_development_overlap_frames": len(frames) - len(strict_frames),
            "n_source_disjoint_frames": len(strict_frames),
            "n_valid_source_disjoint_frames": len(valid_frames),
            "n_invalid_source_disjoint_frames": len(invalid_strict),
            "invalid_reasons": dict(sorted(invalid_reasons.items())),
            "n_source_disjoint_scenes": len(
                {frame.scene_id for frame in strict_frames}
            ),
            "n_valid_source_disjoint_scenes": len(
                {frame.scene_id for frame in valid_frames}
            ),
        },
        "coverage": _coverage_metadata(coverage),
        "discretization": discretization,
        "outcomes": outcomes,
        "representative_orders": representative_orders,
        "limitations": [
            "This is an adaptive exploratory analysis on a previously inspected suite.",
            "Outcomes describe emitted discrete plans rather than executed motion.",
            "Recorded agents do not react to the counterfactual ego plan.",
            "Center off-road is a high-precision proxy that misses partial-footprint departures.",
            "The trajectory arm is mechanically close to outcomes defined from the same plan.",
            "The trajectory-grid baseline measures ego-centric plan shape, not geographic map coverage.",
            "Adjacent positive frames from one scene are dependent, so a long failure contributes once per frame "
            "and intervals understate corpus uncertainty.",
            "Episode boundaries use exact 100 ms adjacency and have not been externally validated.",
            "Reported intervals reflect randomized coverage tie breaking, not sampling uncertainty.",
            "Quantile bins are fit on the valid OOD cohort without reading outcome labels.",
            "Table coverage orders are constructed without reading collision or off-road labels.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate reasoning coverage on prospective plan-safety proxies"
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
        "--labels",
        type=Path,
        default=Path(
            "runs/reasoning_coverage/ood_hazard_2604.planned_safety_3s_100ms.json"
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
        default=Path(
            "runs/reasoning_coverage/ood_hazard_2604.planned_safety_coverage.json"
        ),
    )
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_planned_safety_evaluation(
        args.root,
        args.benchmark,
        args.labels,
        args.cache,
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
    for outcome in report["config"]["primary_outcomes"]:
        result = report["outcomes"][outcome]
        print(f"{outcome}: {result['n_positive_frames']} positive frames")
        for arm in (
            "random",
            "trajectory",
            "trajectory_grid",
            "text_bigram",
            "table_z3_primary",
            "table_z1_x_z3_primary",
            "table_primary",
        ):
            value = result["arms"][arm]["frame_apfd"]["mean"]
            print(f"  {arm:18s} {value:.3f}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
