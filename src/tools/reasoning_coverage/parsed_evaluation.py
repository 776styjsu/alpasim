# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Prioritize the source-disjoint planned-safety cohort using parsed reasoning."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import _flatten_cot
from .experiment_analysis import (
    DatasetAnalysis,
    _budget_count,
    _coverage_report,
    _curve_report,
    _detected_faults,
    _distribution,
    _exclusive_summary,
    _fault_unit_apfd,
)
from .ood_evaluation import _development_scene_ids, lazy_greedy_coverage_order
from .parsed_paths import ARM_NAMES, coverage_capacities, load_parsed_coverage
from .planned_safety_evaluation import _load_label_rows
from .trajectory_grid import (
    DEFAULT_CELL_SIZE_M,
    DEFAULT_HORIZON_S,
    DEFAULT_RASTER_STEP_M,
    trajectory_grid_items_from_points,
)

ARCHIVE = Path(
    "experiments/archive/reasoning_coverage/2026-09-06/runs/reasoning_coverage"
)
BUDGETS = (0.01, 0.05, 0.1, 0.25)
MODES = ("single_round", "multi_round")
OUTCOMES = {
    "collision": "open_loop_collision_with_recorded_agents",
    "offroad": "planned_center_offroad",
    "combined": "open_loop_safety_violation",
}
DISPLAY = {"random": "Random", "trajectory_grid": "Trajectory grid", **ARM_NAMES}
COLORS = {
    "random": "#777777",
    "trajectory_grid": "#e58606",
    "independent": "#5d69b1",
    "additive": "#52bca3",
    "path_single": "#cc61b0",
    "path_le2": "#247ba0",
    "path_all": "#b22222",
    "z3_single": "#24796c",
    "z3_all": "#764e9f",
}
if COLORS.keys() != DISPLAY.keys():
    raise RuntimeError("Every plotted arm needs a colour")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_grid(path: Path) -> tuple[frozenset[str], dict[str, str]]:
    """Hash and rasterize the exact emitted trajectory used by the control arm."""
    info_path = path.with_name("additional_info.json")
    data = info_path.read_bytes()
    info = json.loads(data)
    if (
        info["coordinate_frame"] != "ego_rig_frame"
        or info["trajectory_source"] != "trajectory_xy_rig_frame"
    ):
        raise ValueError(f"Unexpected trajectory coordinate convention: {info_path}")
    points = [
        (
            float(row["relative_time_s"]),
            float(row["position_m"]["x"]),
            float(row["position_m"]["y"]),
        )
        for row in info["waypoints"]
    ]
    items = frozenset(trajectory_grid_items_from_points(points))
    if not items:
        raise ValueError(f"Empty trajectory grid: {info_path}")
    return items, {"path": str(info_path), "sha256": hashlib.sha256(data).hexdigest()}


def _arm_orders(
    case_ids: tuple[str, ...],
    item_sets: dict[str, frozenset[str]],
    repeats: int,
    seed: int,
    multi_round: bool,
) -> list[list[str]]:
    return [
        lazy_greedy_coverage_order(
            case_ids, item_sets, seed + repeat, multi_round=multi_round
        )
        for repeat in range(repeats)
    ]


def _unique_discovery(
    analysis: DatasetAnalysis, frame_context: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare each reasoning method with the grid at equal, paired budgets."""
    report = {}
    budget_rows = []
    fault_rows = []
    for outcome, units in analysis.outcomes.items():
        outcome_report = {}
        fault_ids = set(units)
        for fraction in BUDGETS:
            budget = _budget_count(len(analysis.case_ids), fraction)
            detected = {
                arm: [_detected_faults(order, units, budget) for order in orders]
                for arm, orders in analysis.orders.items()
            }
            probabilities = {}
            for arm, repeats in detected.items():
                counts = Counter(fault for selected in repeats for fault in selected)
                probabilities[arm] = {
                    fault: counts[fault] / len(repeats) for fault in fault_ids
                }
            comparisons = {}
            for candidate in ARM_NAMES:
                summary, exclusive_probability = _exclusive_summary(
                    detected,
                    fault_ids,
                    candidate_arms=(candidate,),
                    comparator_arms=("trajectory_grid",),
                )
                robust = set(summary["robustly_exclusive_faults"])
                grid_robust = {
                    fault
                    for fault in fault_ids
                    if probabilities["trajectory_grid"][fault] >= 0.75
                    and probabilities[candidate][fault] <= 0.25
                }
                summary["robustly_grid_exclusive_faults"] = sorted(grid_robust)
                summary["candidate_detected"] = _distribution(
                    [len(values) for values in detected[candidate]]
                )
                summary["grid_detected"] = _distribution(
                    [len(values) for values in detected["trajectory_grid"]]
                )
                summary["paired_detected_delta"] = _distribution(
                    [
                        len(a) - len(b)
                        for a, b in zip(
                            detected[candidate],
                            detected["trajectory_grid"],
                            strict=True,
                        )
                    ]
                )
                comparisons[candidate] = summary
                budget_rows.append(
                    {
                        "outcome": outcome,
                        "budget_fraction": fraction,
                        "budget_frames": budget,
                        "arm": candidate,
                        "detected_mean": summary["candidate_detected"]["mean"],
                        "recall_mean": summary["candidate_detected"]["mean"]
                        / len(units),
                        "grid_detected_mean": summary["grid_detected"]["mean"],
                        "delta_detected_mean": summary["paired_detected_delta"]["mean"],
                        "delta_detected_p2_5": summary["paired_detected_delta"]["p2_5"],
                        "delta_detected_p97_5": summary["paired_detected_delta"][
                            "p97_5"
                        ],
                        "candidate_only_mean": summary["candidate_only_faults"]["mean"],
                        "grid_only_mean": summary["comparator_only_faults"]["mean"],
                        "overlap_mean": summary["overlapping_faults"]["mean"],
                        "robust_candidate_only": len(robust),
                        "robust_grid_only": len(grid_robust),
                    }
                )
                for fault in sorted(fault_ids):
                    fault_rows.append(
                        {
                            "outcome": outcome,
                            "budget_fraction": fraction,
                            "arm": candidate,
                            "frame_id": fault,
                            **frame_context[fault],
                            "candidate_detection_probability": probabilities[candidate][
                                fault
                            ],
                            "grid_detection_probability": probabilities[
                                "trajectory_grid"
                            ][fault],
                            "candidate_only_probability": exclusive_probability[fault],
                            "robust_candidate_only": fault in robust,
                            "robust_grid_only": fault in grid_robust,
                        }
                    )
            outcome_report[str(fraction)] = comparisons
        report[outcome] = outcome_report
    return report, budget_rows, fault_rows


def _plots(report: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mode in MODES:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
        for ax, outcome in zip(axes, OUTCOMES, strict=True):
            result = report["results"][mode]["outcomes"][outcome]
            for arm, curve in result["curves"].items():
                x = np.asarray(curve["budget_fraction"]) * 100
                y = np.asarray(curve["mean_faults"]) / result["n_faults"] * 100
                ax.plot(
                    x,
                    y,
                    label=DISPLAY[arm],
                    color=COLORS[arm],
                    linestyle="--" if arm == "random" else "-",
                )
            ax.set(
                xlim=(0, 25),
                ylim=(0, 100),
                xlabel="Frames reviewed (%)",
                ylabel="Positive frames detected (%)",
                title=f"{outcome.title()} ({result['n_faults']:,} positive frames)",
            )
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(
            f"{mode.replace('_', ' ').title()} prioritization · "
            f"{report['cohort']['n_frames']:,} frames · {report['config']['repeats']} seeds"
        )
        for extension in ("png", "pdf"):
            fig.savefig(output / f"recall_{mode}.{extension}", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, outcome in zip(axes, ("collision", "offroad"), strict=True):
        rows = report["results"]["multi_round"]["unique_discovery"][outcome]["0.1"]
        x = np.arange(len(ARM_NAMES))
        candidate_only = [
            rows[arm]["candidate_only_faults"]["mean"] for arm in ARM_NAMES
        ]
        grid_only = [rows[arm]["comparator_only_faults"]["mean"] for arm in ARM_NAMES]
        ax.bar(
            x - 0.2, candidate_only, width=0.4, label="Reasoning only", color="#247ba0"
        )
        ax.bar(
            x + 0.2,
            grid_only,
            width=0.4,
            label="Grid only",
            color=COLORS["trajectory_grid"],
        )
        ax.set_xticks(x, [ARM_NAMES[arm] for arm in ARM_NAMES], rotation=20, ha="right")
        ax.set(ylabel="Mean exclusive positive frames", title=outcome.title())
        ax.legend()
    fig.suptitle("Same-budget discoveries at 10% · multi-round · paired tie seeds")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"unique_at_10pct.{extension}", dpi=180)
    plt.close(fig)


def _markdown(report: dict[str, Any]) -> str:
    cohort = report["cohort"]
    lines = [
        "# Parsed reasoning coverage: large-cohort prioritization",
        "",
        f"Evaluated **{cohort['n_frames']:,} valid frames from {cohort['n_scenes']} source-disjoint scenes**, "
        f"with {cohort['n_unique_reasoning_texts']:,} unique parsed texts. "
        "Every eligible frame has a parse and trajectory.",
        "",
        "The seven reasoning arms use the agreed independent, fully additive, single-field path, ≤2-field path, "
        "all-partial-path, and proposition-only (Z3) item definitions. Missing modifiers earn no complete "
        "decision/path coverage; empty item sets remain in the cohort. All parsed statuses and timings are "
        "included. Independent arms count unlinked mentions. The Z3 arms drop the decision and role axes, so "
        "they score propositions without requiring a complete decision or a role link. Fixed capacities are "
        "reported but do not change marginal-gain rankings.",
        "",
        f"All arms use {report['config']['repeats']} shared seeds starting at {report['config']['seed']}. "
        "Multi-round greedy resets the covered set at saturation; single-round appends the seeded random tail. "
        "Both retain a seeded tail once only empty item sets remain. Coverage ordering never uses fault labels.",
        "",
        "Trajectory-grid coverage uses the emitted first 3 seconds in the ego rig frame, origin-centered 2 m cells, "
        "and interpolation samples at most 0.5 m apart. Outcomes are the archived open-loop collision and "
        "ego-center-off-road proxies, plus their union. One positive frame is one fault unit.",
        "",
        "## Multi-round APFD and recall at 10%",
        "",
        "| Method | Collision APFD | Collision recall | Off-road APFD | Off-road recall | Union APFD | Union recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    primary = report["results"]["multi_round"]
    for arm in DISPLAY:
        values = []
        for outcome in OUTCOMES:
            row = primary["outcomes"][outcome]
            values += [
                f"{row['apfd'][arm]['mean']:.4f}",
                f"{100 * row['at_budget']['0.1'][arm]['mean_recall']:.2f}%",
            ]
        lines.append(f"| {DISPLAY[arm]} | " + " | ".join(values) + " |")
    lines += [
        "",
        "![Recall curves](recall_multi_round.png)",
        "",
        "## Unique discoveries versus trajectory grid at 10%",
        "",
        "Each arm receives the same frame budget. Candidate-only and grid-only counts "
        "are averaged across paired seeds. "
        "Robust exclusivity means candidate detection probability ≥0.75 and grid probability ≤0.25. "
        "These are unique-to-an-ordering frames, not distinct physical incidents or root causes.",
        "",
        "| Method | Collision: reasoning only | Collision: grid only | Collision: robust reasoning only | "
        "Off-road: reasoning only | Off-road: grid only | Off-road: robust reasoning only |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARM_NAMES:
        values = []
        for outcome in ("collision", "offroad"):
            row = primary["unique_discovery"][outcome]["0.1"][arm]
            values += [
                f"{row['candidate_only_faults']['mean']:.2f}",
                f"{row['comparator_only_faults']['mean']:.2f}",
                str(len(row["robustly_exclusive_faults"])),
            ]
        lines.append(f"| {DISPLAY[arm]} | " + " | ".join(values) + " |")
    lines += [
        "",
        "![Exclusive discoveries](unique_at_10pct.png)",
        "",
        "## Coverage and first-round saturation",
        "",
        "| Method | Observed items | Fixed capacity | Frames with empty coverage | "
        "Median frames to cover observed items |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, row in report["coverage_audit"].items():
        capacity = row["fixed_capacity"]
        lines.append(
            f"| {DISPLAY[arm]} | {row['observed_items']} | {capacity if capacity is not None else 'cohort-relative'} | "
            f"{row['empty_frames']} | {primary['coverage'][arm]['saturation_cases']['median']:.1f} |"
        )
    lines += [
        "",
        "## Single-round sensitivity",
        "",
        "| Method | Collision APFD | Off-road APFD | Union APFD |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in DISPLAY:
        values = [
            f"{report['results']['single_round']['outcomes'][outcome]['apfd'][arm]['mean']:.4f}"
            for outcome in OUTCOMES
        ]
        lines.append(f"| {DISPLAY[arm]} | " + " | ".join(values) + " |")
    lines += [
        "",
        "![Single-round recall curves](recall_single_round.png)",
        "",
        "## Interpretation limits and artifacts",
        "",
        "Intervals in the machine-readable results are 2.5–97.5 percentiles across tie-breaking seeds, "
        "not confidence intervals over new scenes. Adjacent frames are dependent, so long failures contribute "
        "many positives. The CoT parser has not received an independent human accuracy audit. "
        "The cohort is source-disjoint from development, but these are retrospective results, not a new holdout study.",
        "",
        "- [Full results](summary.json), including paired APFD and detection deltas versus grid.",
        "- [APFD table](apfd.csv) and [budget and exclusivity table](budget_metrics.csv).",
        "- [Per-fault detection probabilities](fault_probabilities.csv), with frame, scene, and parsed-text IDs.",
        "- [Cohort and outcomes](cohort.csv) and [input/configuration manifest](manifest.json).",
        "- `coverage_sets.json.gz` preserves per-frame items; `orders.npz` preserves "
        "every order as indices into `frame_ids.json`.",
        "- Plot PDFs accompany the PNGs for export.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=src/tools:src/utils uv run --no-sync python -m reasoning_coverage.parsed_evaluation \\",
        f"  --output-dir {report['config']['output_dir']}_rerun \\",
        f"  --repeats {report['config']['repeats']} --seed {report['config']['seed']} "
        f"--workers {report['config']['workers']}",
        "```",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeats <= 0 or args.workers <= 0:
        raise ValueError("repeats and workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schema = json.loads(args.schema.read_text())
    capacities = coverage_capacities(schema)
    parsed, parse_ids = load_parsed_coverage(
        args.parse_inputs, args.parse_outputs, schema
    )
    features = json.loads(args.features.read_text())
    if features["schema_version"] != 2:
        raise ValueError("Unexpected feature-cache version")
    _, label_rows = _load_label_rows(args.labels)
    development = _development_scene_ids(args.benchmark)
    frames = {}
    for raw, path in zip(features["frames"], features["metadata_paths"], strict=True):
        frame_id = raw["frame_id"]
        if Path(path).parent.name != frame_id or frame_id in frames:
            raise ValueError(f"Feature/metadata alignment error: {frame_id}")
        frames[frame_id] = {**raw, "metadata_path": Path(path)}
    nonoverlap = {
        key: frame
        for key, frame in frames.items()
        if frame["scene_id"] not in development
    }
    cohort = {
        key: frame for key, frame in nonoverlap.items() if label_rows[key]["valid"]
    }
    case_ids = tuple(sorted(cohort))
    if not case_ids:
        raise ValueError("Empty source-disjoint valid cohort")
    text_by_frame = {
        key: _flatten_cot(frame["chain_of_thought"]) for key, frame in cohort.items()
    }
    missing = {text for text in text_by_frame.values() if text not in parsed}
    if missing:
        raise ValueError(
            f"Missing parses for {len(missing)} cohort texts; examples: {sorted(missing)[:3]}"
        )
    coverage = {
        arm: {frame_id: parsed[text_by_frame[frame_id]][arm] for frame_id in case_ids}
        for arm in ARM_NAMES
    }
    print(
        f"Mapped all {len(case_ids):,} frames to {len(set(text_by_frame.values())):,} "
        "parsed texts; loading grid trajectories.",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        grids = list(
            pool.map(_load_grid, (cohort[key]["metadata_path"] for key in case_ids))
        )
    coverage = {
        "trajectory_grid": {
            key: result[0] for key, result in zip(case_ids, grids, strict=True)
        },
        **coverage,
    }
    source_paths = [
        args.schema,
        args.benchmark,
        args.features,
        args.labels,
        *sorted(args.parse_inputs.glob("*.json")),
        *sorted(args.parse_outputs.glob("*.jsonl")),
    ]
    manifest = {
        "inputs": [
            {"path": str(path.resolve()), "sha256": _file_hash(path)}
            for path in source_paths
        ],
        "trajectories": [result[1] for result in grids],
        "implementation": [
            {"path": str(path.resolve()), "sha256": _file_hash(path)}
            for path in (
                Path(__file__),
                Path(__file__).with_name("parsed_paths.py"),
                Path(__file__).with_name("ood_evaluation.py"),
                Path(__file__).with_name("experiment_analysis.py"),
                Path(__file__).with_name("trajectory_grid.py"),
            )
        ],
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    _write_json(args.output_dir / "frame_ids.json", case_ids)
    with gzip.open(args.output_dir / "coverage_sets.json.gz", "wt") as file:
        json.dump(
            {
                arm: {key: sorted(items) for key, items in sets.items()}
                for arm, sets in coverage.items()
            },
            file,
        )
    coverage_audit = {}
    for arm, items in coverage.items():
        universe = frozenset().union(*items.values())
        capacity = None if arm == "trajectory_grid" else capacities[arm]
        coverage_audit[arm] = {
            "observed_items": len(universe),
            "fixed_capacity": capacity,
            "fixed_coverage": None if capacity is None else len(universe) / capacity,
            "empty_frames": sum(not value for value in items.values()),
            "unique_item_sets": len(set(items.values())),
        }
    print(
        "Coverage prepared: "
        + ", ".join(
            f"{arm}={row['observed_items']} items"
            for arm, row in coverage_audit.items()
        ),
        flush=True,
    )

    random_orders = []
    for repeat in range(args.repeats):
        order = list(case_ids)
        random.Random(args.seed + repeat).shuffle(order)
        random_orders.append(order)
    orders = {mode: {"random": random_orders} for mode in MODES}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {
            pool.submit(
                _arm_orders,
                case_ids,
                items,
                args.repeats,
                args.seed,
                mode == "multi_round",
            ): (mode, arm)
            for mode in MODES
            for arm, items in coverage.items()
        }
        for task in as_completed(tasks):
            mode, arm = tasks[task]
            orders[mode][arm] = task.result()
            print(f"Orders complete: {mode} / {arm}", flush=True)
    index = {frame_id: n for n, frame_id in enumerate(case_ids)}
    arrays = {}
    for mode in MODES:
        orders[mode] = {arm: orders[mode][arm] for arm in DISPLAY}
        for arm, repeats in orders[mode].items():
            array = np.asarray(
                [[index[key] for key in order] for order in repeats], dtype=np.int32
            )
            if array.shape != (args.repeats, len(case_ids)) or not np.all(
                np.sort(array, axis=1) == np.arange(len(case_ids))
            ):
                raise ValueError(f"Invalid ordering permutation: {mode}/{arm}")
            arrays[f"{mode}__{arm}"] = array
    np.savez_compressed(args.output_dir / "orders.npz", **arrays)
    print(
        "All outcome-blind orders saved; joining fault outcomes and computing metrics.",
        flush=True,
    )

    outcomes = {
        outcome: {key: frozenset((key,)) for key in case_ids if label_rows[key][field]}
        for outcome, field in OUTCOMES.items()
    }
    if any(not units for units in outcomes.values()):
        raise ValueError("An outcome has no positive frames")
    frame_context = {
        key: {
            "scene_id": cohort[key]["scene_id"],
            "trace_id": parse_ids[text_by_frame[key]],
            "time_now_us": label_rows[key]["time_now_us"],
        }
        for key in case_ids
    }
    _write_csv(
        args.output_dir / "cohort.csv",
        [
            {
                "frame_id": key,
                **frame_context[key],
                **{name: int(key in units) for name, units in outcomes.items()},
            }
            for key in case_ids
        ],
    )
    report = {
        "config": manifest["config"],
        "cohort": {
            "source_frames": len(frames),
            "source_disjoint_frames": len(nonoverlap),
            "excluded_development_frames": len(frames) - len(nonoverlap),
            "excluded_invalid_frames": len(nonoverlap) - len(cohort),
            "n_frames": len(cohort),
            "n_scenes": len({frame["scene_id"] for frame in cohort.values()}),
            "n_unique_reasoning_texts": len(set(text_by_frame.values())),
            "positive_frames": {name: len(units) for name, units in outcomes.items()},
        },
        "coverage_audit": coverage_audit,
        "trajectory_grid": {
            "cell_size_m": DEFAULT_CELL_SIZE_M,
            "horizon_s": DEFAULT_HORIZON_S,
            "raster_step_m": DEFAULT_RASTER_STEP_M,
        },
        "policy": {
            "fault_unit": "one positive frame; no episode merging",
            "parser_scope": "all statuses, timings, assertions; unlinked mentions included for independent arms",
            "missing_values": "no coverage credit; incomplete decisions excluded from variant/path arms only",
            "unique_faults": "same-budget pairwise positive-frame set difference against trajectory_grid",
            "robust_unique": "candidate detection probability >=0.75 and grid probability <=0.25 across tie seeds",
            "intervals": (
                "2.5–97.5 percentiles over seeded tie-breaking repeats, "
                "not scene-population confidence intervals"
            ),
        },
        "results": {},
    }
    apfd_rows, all_budget_rows, all_fault_rows, all_curve_rows = [], [], [], []
    for mode in MODES:
        analysis = DatasetAnalysis(case_ids, coverage, orders[mode], outcomes)
        curves, curve_rows = _curve_report(
            analysis, OUTCOMES, points=200, value_budgets=BUDGETS
        )
        saturation = _coverage_report(analysis, points=200)
        unique, budget_rows, fault_rows = _unique_discovery(analysis, frame_context)
        deltas = {}
        for outcome, units in outcomes.items():
            values = {
                arm: [_fault_unit_apfd(order, units) for order in repeats]
                for arm, repeats in orders[mode].items()
            }
            deltas[outcome] = {}
            for arm in DISPLAY:
                delta = _distribution(
                    [
                        a - b
                        for a, b in zip(
                            values[arm], values["trajectory_grid"], strict=True
                        )
                    ]
                )
                deltas[outcome][arm] = delta
                row = curves[outcome]["apfd"][arm]
                apfd_rows.append(
                    {
                        "mode": mode,
                        "outcome": outcome,
                        "arm": arm,
                        "positive_frames": len(units),
                        **row,
                        **{
                            f"delta_vs_grid_{key}": value
                            for key, value in delta.items()
                        },
                    }
                )
        report["results"][mode] = {
            "outcomes": curves,
            "coverage": saturation,
            "unique_discovery": unique,
            "paired_apfd_delta_vs_grid": deltas,
        }
        all_budget_rows.extend({"mode": mode, **row} for row in budget_rows)
        all_fault_rows.extend({"mode": mode, **row} for row in fault_rows)
        all_curve_rows.extend({"mode": mode, **row} for row in curve_rows)
        print(f"Metrics complete: {mode}", flush=True)
    _write_json(args.output_dir / "summary.json", report)
    for name, rows in (
        ("apfd.csv", apfd_rows),
        ("budget_metrics.csv", all_budget_rows),
        ("fault_probabilities.csv", all_fault_rows),
        ("curves.csv", all_curve_rows),
    ):
        _write_csv(args.output_dir / name, rows)
    _plots(report, args.output_dir)
    (args.output_dir / "REPORT.md").write_text(_markdown(report))
    print(f"Report complete: {args.output_dir / 'REPORT.md'}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("data/reasoning_coverage/decision_role_proposition_v1.json"),
    )
    parser.add_argument(
        "--parse-inputs", type=Path, default=Path("experiments/cot_parse/batches")
    )
    parser.add_argument(
        "--parse-outputs", type=Path, default=Path("experiments/cot_parse/out")
    )
    parser.add_argument(
        "--features", type=Path, default=ARCHIVE / "ood_hazard_2604.features.json"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ARCHIVE / "ood_hazard_2604.planned_safety_3s_100ms.json",
    )
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
