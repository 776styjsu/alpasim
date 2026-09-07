# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Run the retrospective Alpamayo reasoning-coverage pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import build_cases, run_pilot, write_curve_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Alpamayo reasoning coverage with non-reasoning baselines"
    )
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark.json"))
    parser.add_argument(
        "--audit-labels",
        type=Path,
        default=None,
        help="Optional reviewed reasoning-label overlays keyed by clip_id",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/reasoning_coverage/benchmark_pilot.json"),
    )
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=Path("runs/reasoning_coverage/benchmark_pilot.curves.csv"),
    )
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases, artifact_counts = build_cases(args.benchmark, args.audit_labels)
    report = run_pilot(cases, bins=args.bins, repeats=args.repeats, seed=args.seed)
    report["inputs"] = {
        "benchmark": str(args.benchmark),
        "audit_labels": str(args.audit_labels) if args.audit_labels else None,
        "artifact_counts": artifact_counts,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    write_curve_csv(report, args.curve_csv)

    print(f"Cases: {report['dataset']['n_cases']}")
    print(f"Outcome counts: {report['dataset']['outcome_counts']}")
    print("APFD for any_issue (mean over randomized tie breaks):")
    for arm, summary in report["arms"].items():
        score = summary["outcomes"]["any_issue"]["apfd"]
        rendered = "n/a" if score is None else f"{score['mean']:.3f}"
        print(f"  {arm:18s} {rendered}")
    print(f"Report: {args.output}")
    print(f"Curves: {args.curve_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
