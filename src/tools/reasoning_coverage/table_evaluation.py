# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Compute Alpamayo table-program coverage on IID cases and individual frames."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

import numpy as np

from .alpamayo_tables import (
    TABLE1_LATERAL_DECISIONS,
    TABLE1_LONGITUDINAL_DECISIONS,
    TABLE2_COMPONENT_CATEGORIES,
    TABLE5_LATERAL_ACTIONS,
    TABLE5_LONGITUDINAL_ACTIONS,
    table1_decision_pairs,
    table2_category_x_table1_decision_pairs,
    table5_action_pairs,
)
from .analysis import apfd, build_cases, load_benchmark
from .table_program import (
    CAUSAL_ROLES,
    AtomicActionRun,
    TableReasoningProgram,
    action_items,
    compile_table_program,
    load_action_runs,
    primary_reasoning_items,
    reasoning_action_items,
    sequence_reasoning_items,
    z1_category_x_z3_pair_items,
    z3_pair_items,
)
from .trajectory_grid import trajectory_grid_items

ARMS = (
    "random",
    "trajectory_grid",
    "table_z3_primary",
    "table_z3_sequence",
    "table_z1_x_z3_primary",
    "table_z1_x_z3_sequence",
    "table_primary",
    "table_sequence",
    "table_action",
    "table_reasoning_action",
)


@dataclass(frozen=True)
class CoverageRecord:
    case_id: str
    source_group: str
    program: TableReasoningProgram
    action_runs: tuple[AtomicActionRun, ...]
    outcomes: tuple[tuple[str, bool], ...]
    trajectory_grid_items: tuple[str, ...] = ()

    def outcome(self, name: str) -> bool:
        return dict(self.outcomes).get(name, False)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("table_program.py"),
        Path(__file__).with_name("alpamayo_tables.py"),
        Path(__file__).with_name("program.py"),
    ):
        digest.update(path.name.encode())
        digest.update(_file_sha256(path).encode())
    return digest.hexdigest()


def _frame_collection_fingerprint(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for metadata_path in paths:
        for path in (metadata_path, metadata_path.with_name("additional_info.json")):
            stat = path.stat()
            digest.update(str(path.relative_to(root)).encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _resolve_metadata_path(entry: dict[str, Any], benchmark_path: Path) -> Path | None:
    raw_path = entry.get("source_scene_file")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    root = benchmark_path.resolve().parent
    candidates = [path] if path.is_absolute() else []
    candidates.extend((root / path, root / "tutorial_alpamayo" / path))
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _base_clip_id(frame_id: str) -> str:
    return frame_id.rsplit("_t", maxsplit=1)[0]


def _program_cache(texts: Iterable[str]) -> dict[str, TableReasoningProgram]:
    return {text: compile_table_program(text) for text in sorted(set(texts))}


def load_benchmark_records(
    benchmark_path: Path,
) -> tuple[list[CoverageRecord], dict[str, Any]]:
    cases, artifact_counts = build_cases(benchmark_path)
    entries = {str(entry["clip_id"]): entry for entry in load_benchmark(benchmark_path)}
    programs = _program_cache(case.chain_of_thought for case in cases)
    records: list[CoverageRecord] = []
    action_artifacts = 0
    for case in cases:
        metadata_path = _resolve_metadata_path(entries[case.clip_id], benchmark_path)
        runs = load_action_runs(metadata_path)
        action_artifacts += bool(runs)
        records.append(
            CoverageRecord(
                case_id=case.clip_id,
                source_group=_base_clip_id(case.clip_id),
                program=programs[case.chain_of_thought],
                action_runs=runs,
                outcomes=case.outcomes,
                trajectory_grid_items=case.trajectory_grid_items,
            )
        )
    return records, {
        "artifact_counts": {
            **artifact_counts,
            "full_action_sequences": action_artifacts,
        },
        "n_unique_reasoning_texts": len(programs),
    }


def _load_frame_input(
    metadata_path: Path,
) -> tuple[str, str, tuple[AtomicActionRun, ...], tuple[str, ...]]:
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    frame_id = str(metadata["clip_id"])
    if frame_id != metadata_path.parent.name:
        raise ValueError(
            f"Frame ID mismatch: {frame_id} != {metadata_path.parent.name}"
        )
    text = str(metadata.get("chain_of_thought", ""))
    return (
        frame_id,
        text,
        load_action_runs(metadata_path),
        tuple(sorted(trajectory_grid_items(metadata_path))),
    )


def _unsafe_frame_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("entries", payload.get("results"))
    else:
        entries = None
    if not isinstance(entries, list):
        raise ValueError(f"Expected unsafe-event entries in {path}")
    return {str(entry["clip_id"]) for entry in entries if isinstance(entry, dict)}


def load_frame_records(
    root: Path, unsafe_path: Path, workers: int
) -> tuple[list[CoverageRecord], dict[str, Any]]:
    paths = sorted(root.rglob("metadata.json"), key=str)
    if not paths:
        raise ValueError(f"No metadata.json files under {root}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        loaded = list(executor.map(_load_frame_input, paths))
    frame_ids = [frame_id for frame_id, _, _, _ in loaded]
    duplicates = [item for item, count in Counter(frame_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate frame IDs: {sorted(duplicates)}")
    programs = _program_cache(text for _, text, _, _ in loaded)
    unsafe_ids = _unsafe_frame_ids(unsafe_path)
    known_unsafe = unsafe_ids & set(frame_ids)
    records = [
        CoverageRecord(
            case_id=frame_id,
            source_group=_base_clip_id(frame_id),
            program=programs[text],
            action_runs=runs,
            outcomes=(("known_manual_unsafe_event", frame_id in known_unsafe),),
            trajectory_grid_items=grid_items,
        )
        for frame_id, text, runs, grid_items in loaded
    ]
    return records, {
        "root": str(root),
        "n_metadata_files": len(paths),
        "frame_collection_fingerprint": _frame_collection_fingerprint(paths, root),
        "n_unique_reasoning_texts": len(programs),
        "n_action_sequences": sum(bool(record.action_runs) for record in records),
        "n_supplied_unsafe_events": len(unsafe_ids),
        "n_matched_unsafe_events": len(known_unsafe),
    }


def build_coverage_sets(
    records: list[CoverageRecord],
) -> dict[str, dict[str, frozenset[str]]]:
    coverage = {arm: {} for arm in ARMS if arm != "random"}
    for record in records:
        values = {
            "trajectory_grid": set(record.trajectory_grid_items),
            "table_z3_primary": z3_pair_items(record.program),
            "table_z3_sequence": z3_pair_items(
                record.program, include_continuations=True
            ),
            "table_z1_x_z3_primary": z1_category_x_z3_pair_items(record.program),
            "table_z1_x_z3_sequence": z1_category_x_z3_pair_items(
                record.program, include_continuations=True
            ),
            "table_primary": primary_reasoning_items(record.program),
            "table_sequence": sequence_reasoning_items(record.program),
            "table_action": action_items(record.action_runs),
            "table_reasoning_action": reasoning_action_items(
                record.program, record.action_runs
            ),
        }
        for arm, items in values.items():
            coverage[arm][record.case_id] = frozenset(items)
    return coverage


def _dimension(values: set[str], denominator: tuple[str, ...]) -> dict[str, Any]:
    denominator_set = set(denominator)
    unknown = values - denominator_set
    if unknown:
        raise ValueError(f"Values outside fixed denominator: {sorted(unknown)}")
    return {
        "covered": len(values),
        "total": len(denominator),
        "fraction": len(values) / len(denominator),
        "missing": sorted(denominator_set - values),
    }


def fixed_denominator_coverage(records: list[CoverageRecord]) -> dict[str, Any]:
    primary_stages = [record.program.primary for record in records]
    all_stages = [stage for record in records for stage in record.program.stages]

    def reasoning_dimensions(stages: list[Any]) -> dict[str, Any]:
        categories = {
            component.category for stage in stages for component in stage.components
        }
        roles = {
            clause.role
            for stage in stages
            for clause in stage.clauses
            if clause.role in CAUSAL_ROLES
        }
        longitudinal = {stage.decision.longitudinal for stage in stages}
        lateral = {stage.decision.lateral for stage in stages}
        pairs = {
            stage.decision.pair
            for stage in stages
            if stage.decision.pair != "none|none"
        }
        category_x_pair = {
            f"{component.category}|{stage.decision.pair}"
            for stage in stages
            for component in stage.components
            if stage.decision.pair != "none|none"
        }
        return {
            "table2_component_categories": _dimension(
                categories, TABLE2_COMPONENT_CATEGORIES
            ),
            "causal_roles": _dimension(roles, CAUSAL_ROLES),
            "table1_longitudinal_decisions": _dimension(
                longitudinal, TABLE1_LONGITUDINAL_DECISIONS
            ),
            "table1_lateral_decisions": _dimension(lateral, TABLE1_LATERAL_DECISIONS),
            "table1_decision_pairs": _dimension(pairs, table1_decision_pairs()),
            "table2_category_x_table1_decision_pairs": _dimension(
                category_x_pair, table2_category_x_table1_decision_pairs()
            ),
            "open_component_predicates_observed": len(
                {
                    component.predicate
                    for stage in stages
                    for component in stage.components
                }
            ),
        }

    runs = [run for record in records for run in record.action_runs]
    return {
        "primary_reasoning": reasoning_dimensions(primary_stages),
        "continuation_aware_reasoning": reasoning_dimensions(all_stages),
        "table5_action": {
            "longitudinal_actions": _dimension(
                {run.longitudinal for run in runs}, TABLE5_LONGITUDINAL_ACTIONS
            ),
            "lateral_actions": _dimension(
                {run.lateral for run in runs}, TABLE5_LATERAL_ACTIONS
            ),
            "action_pairs": _dimension(
                {run.pair for run in runs}, table5_action_pairs()
            ),
        },
    }


def _lazy_greedy_order(
    case_ids: list[str], item_sets: dict[str, frozenset[str]], seed: int
) -> list[str]:
    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)
    tie_order = {case_id: index for index, case_id in enumerate(shuffled)}
    heap = [
        (-len(item_sets[case_id]), tie_order[case_id], case_id) for case_id in shuffled
    ]
    heapq.heapify(heap)
    covered: set[str] = set()
    selected: list[str] = []
    selected_set: set[str] = set()
    while heap:
        negative_gain, tie, case_id = heapq.heappop(heap)
        if case_id in selected_set:
            continue
        actual_gain = len(item_sets[case_id] - covered)
        if actual_gain != -negative_gain:
            heapq.heappush(heap, (-actual_gain, tie, case_id))
            continue
        if actual_gain == 0:
            selected.extend(
                candidate for candidate in shuffled if candidate not in selected_set
            )
            break
        selected.append(case_id)
        selected_set.add(case_id)
        covered.update(item_sets[case_id])
    return selected


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(fmean(values)),
        "std": float(pstdev(values)),
        "p2_5": float(np.quantile(array, 0.025)),
        "p97_5": float(np.quantile(array, 0.975)),
    }


def _recall(order: list[str], failures: set[str], fraction: float) -> float:
    budget = max(1, math.ceil(len(order) * fraction))
    return len(set(order[:budget]) & failures) / len(failures)


def _coverage_recall(
    order: list[str], item_sets: dict[str, frozenset[str]], fraction: float
) -> float:
    budget = max(1, math.ceil(len(order) * fraction))
    universe = set().union(*item_sets.values())
    if not universe:
        return 0.0
    covered = set().union(*(item_sets[case_id] for case_id in order[:budget]))
    return len(covered) / len(universe)


def _order_digest(order: list[str]) -> str:
    return hashlib.sha256("\n".join(order).encode()).hexdigest()


def evaluate_prioritization(
    records: list[CoverageRecord],
    coverage: dict[str, dict[str, frozenset[str]]],
    *,
    repeats: int,
    seed: int,
    include_representative_orders: bool,
) -> dict[str, Any]:
    case_ids = sorted(record.case_id for record in records)
    outcome_names = sorted({name for record in records for name, _ in record.outcomes})
    failures = {
        name: {record.case_id for record in records if record.outcome(name)}
        for name in outcome_names
    }
    orders: dict[str, list[list[str]]] = {arm: [] for arm in ARMS}
    for repeat in range(repeats):
        repeat_seed = seed + repeat
        random_order = list(case_ids)
        random.Random(repeat_seed).shuffle(random_order)
        orders["random"].append(random_order)
        for arm in ARMS:
            if arm != "random":
                orders[arm].append(
                    _lazy_greedy_order(case_ids, coverage[arm], repeat_seed)
                )

    arms: dict[str, Any] = {}
    for arm, arm_orders in orders.items():
        if arm == "random":
            coverage_summary = None
        else:
            support = Counter(
                item for items in coverage[arm].values() for item in items
            )
            coverage_summary = {
                "n_items": len(support),
                "mean_items_per_case": float(
                    fmean(len(items) for items in coverage[arm].values())
                ),
                "singleton_items": sum(count == 1 for count in support.values()),
                "mean_recall_at_budget": {
                    str(fraction): float(
                        fmean(
                            _coverage_recall(order, coverage[arm], fraction)
                            for order in arm_orders
                        )
                    )
                    for fraction in (0.05, 0.1, 0.25, 0.5)
                },
            }
        outcome_summary: dict[str, Any] = {}
        for name, failing_ids in failures.items():
            if not failing_ids:
                continue
            scores = [float(apfd(order, failing_ids)) for order in arm_orders]
            outcome_summary[name] = {
                "n_failures": len(failing_ids),
                "apfd": _distribution(scores),
                "mean_recall_at_budget": {
                    str(fraction): float(
                        fmean(
                            _recall(order, failing_ids, fraction)
                            for order in arm_orders
                        )
                    )
                    for fraction in (0.05, 0.1, 0.25, 0.5)
                },
            }
        representative = arm_orders[0]
        arms[arm] = {
            "coverage": coverage_summary,
            "outcomes": outcome_summary,
            "representative_order_sha256": _order_digest(representative),
            **(
                {"representative_order": representative}
                if include_representative_orders
                else {}
            ),
        }
    return {
        "repeats": repeats,
        "seed": seed,
        "outcome_counts": {name: len(values) for name, values in failures.items()},
        "arms": arms,
    }


def _program_diagnostics(records: list[CoverageRecord]) -> dict[str, Any]:
    unique = {record.program.text: record.program for record in records}
    stages = [stage for program in unique.values() for stage in program.stages]
    return {
        "n_unique_programs": len(unique),
        "n_unique_programs_with_continuations": sum(
            bool(program.continuations) for program in unique.values()
        ),
        "n_stages": len(stages),
        "n_continuation_stages": sum(
            len(program.continuations) for program in unique.values()
        ),
        "n_none_none_stages": sum(
            stage.decision.pair == "none|none" for stage in stages
        ),
        "n_unobserved_support_clauses": sum(
            clause.premise is None for stage in stages for clause in stage.clauses
        ),
        "n_records_with_action_sequence": sum(
            bool(record.action_runs) for record in records
        ),
    }


def build_report(
    records: list[CoverageRecord],
    *,
    kind: str,
    inputs: dict[str, Any],
    repeats: int,
    seed: int,
    include_representative_orders: bool,
) -> dict[str, Any]:
    coverage = build_coverage_sets(records)
    return {
        "schema_version": 1,
        "kind": kind,
        "inputs": {**inputs, "implementation_sha256": _dependency_sha256()},
        "config": {"repeats": repeats, "seed": seed},
        "dataset": {"n_cases": len(records)},
        "program_diagnostics": _program_diagnostics(records),
        "fixed_denominator_coverage": fixed_denominator_coverage(records),
        "prioritization": evaluate_prioritization(
            records,
            coverage,
            repeats=repeats,
            seed=seed,
            include_representative_orders=include_representative_orders,
        ),
        "semantics": {
            "trajectory_grid": (
                "Occupied 2 m ego-centric cells along the first 3 seconds of the "
                "emitted trajectory."
            ),
            "table_z3_primary": "Immediate non-empty Table 1 decision pair only.",
            "table_z3_sequence": "Non-empty Table 1 decision pairs across explicit stages.",
            "table_z1_x_z3_primary": "Published Table 2 category by immediate Table 1 decision pair products only.",
            "table_z1_x_z3_sequence": "Published Table 2 category by Table 1 decision pair "
            "products across explicit stages.",
            "table_primary": "Z1/Z2/Z3 for the immediate first decision only.",
            "table_sequence": "Primary reasoning plus explicitly verbalized continuation stages.",
            "table_action": "Full trajectory-derived Table 5 action sequence; not reasoning.",
            "table_reasoning_action": "Sequence reasoning plus A and Z3/A co-occurrence items; "
            "not a faithfulness verdict.",
        },
        "limitations": [
            "The deterministic text compiler is not a human-validated semantic parser.",
            "Table 2 is open-ended; only its seven categories have a fixed denominator.",
            "Causal roles are a local typed extension, not an Alpamayo paper vocabulary.",
            "Z3/A co-occurrence does not establish temporal alignment or mechanistic faithfulness.",
            "Table 5 actions and thresholds are read from locally derived trajectory artifacts.",
            "Coverage construction does not read outcome labels; outcomes are used only after ordering.",
        ],
    }


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Alpamayo Table 1/2/5 reasoning-program coverage"
    )
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark.json"))
    parser.add_argument(
        "--frame-root",
        type=Path,
        default=Path(
            "/project/AV/drive-the-thoughts/alpasim/tutorial_alpamayo_ood_hazard_2604"
        ),
    )
    parser.add_argument(
        "--unsafe-events",
        type=Path,
        default=Path(
            "/project/AV/drive-the-thoughts/data/benchmark/"
            "ood_hazard_2604_unsafe_traces.json"
        ),
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path(
            "runs/reasoning_coverage/benchmark150.table_program_coverage.json"
        ),
    )
    parser.add_argument(
        "--frame-output",
        type=Path,
        default=Path(
            "runs/reasoning_coverage/ood_hazard_2604.frames.table_program_coverage.json"
        ),
    )
    parser.add_argument("--benchmark-repeats", type=int, default=200)
    parser.add_argument("--frame-repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.benchmark_repeats <= 0 or args.frame_repeats <= 0:
        raise ValueError("Repeat counts must be positive")

    benchmark_records, benchmark_audit = load_benchmark_records(args.benchmark)
    benchmark_report = build_report(
        benchmark_records,
        kind="alpamayo_table_program_benchmark150_coverage",
        inputs={
            "benchmark": str(args.benchmark),
            "benchmark_sha256": _file_sha256(args.benchmark),
            **benchmark_audit,
        },
        repeats=args.benchmark_repeats,
        seed=args.seed,
        include_representative_orders=True,
    )
    _write_report(benchmark_report, args.benchmark_output)

    frame_records, frame_audit = load_frame_records(
        args.frame_root, args.unsafe_events, args.workers
    )
    frame_report = build_report(
        frame_records,
        kind="alpamayo_table_program_individual_frame_coverage",
        inputs={
            **frame_audit,
            "unsafe_events": str(args.unsafe_events),
            "unsafe_events_sha256": _file_sha256(args.unsafe_events),
        },
        repeats=args.frame_repeats,
        seed=args.seed,
        include_representative_orders=False,
    )
    _write_report(frame_report, args.frame_output)

    print(f"Benchmark cases: {len(benchmark_records)}")
    print(f"Individual frames: {len(frame_records)}")
    print(f"Benchmark report: {args.benchmark_output}")
    print(f"Frame report: {args.frame_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
