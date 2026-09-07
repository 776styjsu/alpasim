#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Plot the distribution of examples over coverage-equivalent paths.

Each evaluated frame contributes one *path*: the set of 2 m ego-frame grid cells
its emitted 3 s plan crosses (``reasoning_coverage.trajectory_grid``). Frames
that cross exactly the same cell set share a path. The figure is a column chart
over the most frequent paths, with each column split into the faulty examples
sharing that path and the remainder.

Two cohorts are supported. ``planned_safety`` uses the ood_hazard_2604 frames,
where an example is faulty when its planned-safety label reports an open-loop
collision with recorded agents or a planned ego center outside the drivable
area. ``benchmark150`` uses the human-labeled benchmark cases, where an example
is faulty when it carries an inconsistent reliable CoT or an unsafe planned
trajectory.

Examples:
    uv run python tools/plot_trajectory_grid_path_histogram.py
    uv run python tools/plot_trajectory_grid_path_histogram.py \
        --cohort benchmark150
    uv run python tools/plot_trajectory_grid_path_histogram.py \
        --cohort benchmark150 --layout curve --representation reasoning_path
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "tools"))

from reasoning_coverage.analysis import build_cases  # noqa: E402
from reasoning_coverage.ood_evaluation import _base_clip_id, _load_entries  # noqa: E402
from reasoning_coverage.table_program import (  # noqa: E402
    compile_table_program,
    primary_reasoning_items,
)
from reasoning_coverage.trajectory_grid import (  # noqa: E402
    DEFAULT_CELL_SIZE_M,
    DEFAULT_HORIZON_S,
    trajectory_grid_items,
)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dedcd6"
FAULTY_COLOR = "#eb6834"
CLEAN_COLOR = "#2a78d6"


COHORTS = ("planned_safety", "benchmark150")
REPRESENTATIONS = ("trajectory_grid", "reasoning_path")
COHORT_UNITS = {"planned_safety": "frames", "benchmark150": "cases"}
COHORT_FAULT_LABELS = {
    "planned_safety": "Faulty (collision or planned center off-road)",
    "benchmark150": "Faulty (inconsistent reliable CoT or unsafe trajectory)",
}
COHORT_CLEAN_LABELS = {
    "planned_safety": "No planned-safety fault",
    "benchmark150": "No labeled fault",
}
COHORT_TITLES = {
    "planned_safety": (
        "Examples per distinct trajectory-grid path: ood_hazard_2604 frames"
    ),
    "benchmark150": (
        "Examples per distinct trajectory-grid path: 150 human-labeled cases"
    ),
}
COHORT_CURVE_TITLES = {
    "planned_safety": (
        "Trajectory-grid path frequency and fault distribution: ood_hazard_2604 frames"
    ),
    "benchmark150": (
        "Trajectory-grid path frequency and fault distribution: 150 human-labeled cases"
    ),
}
COHORT_CELL_TITLES = {
    "planned_safety": (
        "Trajectory-grid cell coverage and fault density: ood_hazard_2604 frames"
    ),
    "benchmark150": (
        "Trajectory-grid cell coverage and fault density: 150 human-labeled cases"
    ),
}
COHORT_TOP_PATHS = {"planned_safety": 30, "benchmark150": 20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, default="planned_safety")
    parser.add_argument(
        "--layout", choices=("ranked", "curve", "cells"), default="ranked"
    )
    parser.add_argument(
        "--representation", choices=REPRESENTATIONS, default="trajectory_grid"
    )
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark.json"))
    parser.add_argument(
        "--planned-labels",
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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-paths", type=int, default=None)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.layout == "cells" and args.representation != "trajectory_grid":
        parser.error("--layout cells is only defined for trajectory_grid cells")
    if args.representation == "reasoning_path" and (
        args.cohort != "benchmark150" or args.layout != "curve"
    ):
        parser.error(
            "reasoning_path currently supports --cohort benchmark150 --layout curve"
        )
    if args.output is None:
        stem = "trajectory_grid_path_histogram"
        if args.layout == "curve":
            stem = "trajectory_grid_path_curve"
        if args.layout == "cells":
            stem = "trajectory_grid_cell_coverage"
        if args.representation == "reasoning_path":
            stem = "reasoning_coverage_path_curve"
        args.output = Path(f"runs/reasoning_coverage/figures/{stem}.{args.cohort}.png")
    if args.top_paths is None:
        args.top_paths = COHORT_TOP_PATHS[args.cohort]
    return args


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _path_cells(items: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    cells = []
    for item in items:
        x_index, y_index = item.removeprefix("trajectory_grid:cell=").split(",")
        cells.append((int(x_index), int(y_index)))
    return tuple(sorted(cells))


def _path_label(cells: tuple[tuple[int, int], ...]) -> str:
    """Name a path by its forward reach in meters and its lateral extent."""
    forward_m = max(x for x, _ in cells) * DEFAULT_CELL_SIZE_M
    lateral = {y for _, y in cells}
    if lateral == {0}:
        return f"{forward_m:.0f} m"
    left = max(lateral) * DEFAULT_CELL_SIZE_M
    right = min(lateral) * DEFAULT_CELL_SIZE_M
    if right >= 0:
        return f"{forward_m:.0f} m, {left:.0f} left"
    if left <= 0:
        return f"{forward_m:.0f} m, {abs(right):.0f} right"
    return f"{forward_m:.0f} m, \u00b1{max(left, abs(right)):.0f} lat"


def _unique_labels(cell_sets: list[tuple[tuple[int, int], ...]]) -> list[str]:
    """Suffix repeated descriptors so every column names exactly one path."""
    labels = [_path_label(cells) for cells in cell_sets]
    counts = Counter(labels)
    seen: Counter = Counter()
    unique = []
    for label in labels:
        if counts[label] == 1:
            unique.append(label)
            continue
        seen[label] += 1
        unique.append(f"{label} ({chr(ord('a') + seen[label] - 1)})")
    return unique


def _planned_safety_records(args: argparse.Namespace) -> list[dict]:
    """Join each valid, source-disjoint frame to its grid path and fault label."""
    cache = _read_json(args.feature_cache)
    labels = _read_json(args.planned_labels)
    label_by_id = {row["frame_id"]: row for row in labels["frames"]}
    development_scenes = {
        _base_clip_id(str(entry["clip_id"])) for entry in _load_entries(args.benchmark)
    }

    frames = [
        (frame, Path(path))
        for frame, path in zip(cache["frames"], cache["metadata_paths"], strict=True)
        if frame["scene_id"] not in development_scenes
        and label_by_id[frame["frame_id"]]["valid"]
    ]
    if not frames:
        raise ValueError("No valid source-disjoint frames to plot")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        item_sets = list(
            executor.map(lambda entry: trajectory_grid_items(entry[1]), frames)
        )

    records = []
    for (frame, _), items in zip(frames, item_sets, strict=True):
        label = label_by_id[frame["frame_id"]]
        records.append(
            {
                "cells": _path_cells(tuple(sorted(items))),
                "faulty": bool(label["open_loop_collision_with_recorded_agents"])
                or bool(label["planned_center_offroad"]),
            }
        )
    return records


def _benchmark150_records(args: argparse.Namespace) -> list[dict]:
    """Join each benchmark case to its selected coverage signature and fault."""
    cases, _ = build_cases(args.benchmark)
    return [
        {
            "cells": (
                _path_cells(case.trajectory_grid_items)
                if args.representation == "trajectory_grid"
                else tuple(
                    sorted(
                        primary_reasoning_items(
                            compile_table_program(case.chain_of_thought)
                        )
                    )
                )
            ),
            "faulty": case.outcome("inconsistent_reliable_cot")
            or case.outcome("unsafe_trajectory"),
        }
        for case in cases
    ]


def collect_paths(args: argparse.Namespace) -> list[dict]:
    if args.cohort == "benchmark150":
        return _benchmark150_records(args)
    if args.representation != "trajectory_grid":
        raise ValueError("Planned-safety records require trajectory_grid")
    return _planned_safety_records(args)


def build_bins(records: list[dict], top_paths: int) -> tuple[list[dict], dict]:
    """Rank paths by example count and fold the tail into one bin."""
    total: Counter = Counter()
    faulty: Counter = Counter()
    for record in records:
        total[record["cells"]] += 1
        if record["faulty"]:
            faulty[record["cells"]] += 1

    ranked = sorted(total.items(), key=lambda item: (-item[1], item[0]))
    head = ranked[:top_paths]
    tail = ranked[top_paths:]
    # Never split a group of equally frequent paths across the head and the tail:
    # showing an arbitrary few of a tie would misstate how the rest are folded.
    if tail and tail[0][1] == head[-1][1]:
        boundary = head[-1][1]
        kept = [item for item in head if item[1] != boundary]
        if kept:
            head, tail = kept, ranked[len(kept) :]
    labels = _unique_labels([cells for cells, _ in head])
    bins = [
        {
            "label": label,
            "cells": cells,
            "total": count,
            "faulty": faulty[cells],
        }
        for label, (cells, count) in zip(labels, head, strict=True)
    ]
    if tail:
        bins.append(
            {
                "label": f"other ({len(tail):,} paths)",
                "cells": None,
                "total": sum(count for _, count in tail),
                "faulty": sum(faulty[cells] for cells, _ in tail),
            }
        )
    straight_total = sum(
        count for cells, count in ranked if {y for _, y in cells} == {0}
    )
    straight_faulty = sum(
        faulty[cells] for cells, _ in ranked if {y for _, y in cells} == {0}
    )
    summary = {
        "n_examples": len(records),
        "n_faulty": sum(faulty.values()),
        "n_distinct_paths": len(total),
        "n_tail_paths": len(tail),
        "n_singleton_paths": sum(1 for _, count in ranked if count == 1),
        "straight_examples": straight_total,
        "straight_faulty": straight_faulty,
        "lateral_examples": len(records) - straight_total,
        "lateral_faulty": sum(faulty.values()) - straight_faulty,
    }
    return bins, summary


def _rounded_top_bar(
    axes, x: float, width: float, height: float, radius: float, color: str
) -> None:
    """Draw a bar that is square on the baseline and rounded at the data end."""
    radius = max(0.0, min(radius, width / 2.0, height))
    left, right = x - width / 2.0, x + width / 2.0
    vertices = [
        (left, 0.0),
        (left, height - radius),
        (left, height),
        (left + radius, height),
        (right - radius, height),
        (right, height),
        (right, height - radius),
        (right, 0.0),
        (left, 0.0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    axes.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color, lw=0, zorder=3))


def plot(bins: list[dict], summary: dict, cohort: str, output: Path) -> None:
    unit = COHORT_UNITS[cohort]
    totals = [bin_["total"] for bin_ in bins]
    figure, axes = plt.subplots(figsize=(13.5, 6.6))
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    figure.subplots_adjust(left=0.065, right=0.99, top=0.79, bottom=0.26)

    axes.set_xlim(-0.7, len(bins) - 0.3)
    axes.set_ylim(0, max(totals) * 1.08)
    axes.set_xticks(range(len(bins)))

    # Convert the fixed 2 px mark specs into data units for this axes box.
    box = axes.get_window_extent(figure.canvas.get_renderer())
    y_per_px = (axes.get_ylim()[1] - axes.get_ylim()[0]) / box.height
    x_per_px = (axes.get_xlim()[1] - axes.get_xlim()[0]) / box.width
    # Mark specs are written for a 100 dpi surface; scale them to this render.
    px = figure.dpi / 100.0
    gap = 2.0 * px * y_per_px
    radius = 4.0 * px * y_per_px
    width = min(0.62, 24.0 * px * x_per_px)

    for position, bin_ in enumerate(bins):
        _rounded_top_bar(axes, position, width, bin_["total"], radius, CLEAN_COLOR)
        if bin_["faulty"]:
            _rounded_top_bar(
                axes,
                position,
                width,
                max(bin_["faulty"] - gap, bin_["faulty"] / 2.0),
                radius,
                FAULTY_COLOR,
            )

    axes.set_xticklabels(
        [bin_["label"] for bin_ in bins],
        fontsize=8.5,
        color=TEXT_SECONDARY,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axes.set_xlabel(
        f"Trajectory-grid path, ranked by {unit[:-1]} count "
        "(forward reach of the 3 s plan; lateral offset where the path leaves the "
        "ego lane cell)",
        fontsize=9.5,
        color=TEXT_SECONDARY,
        labelpad=8,
    )
    axes.set_ylabel(f"Examples ({unit})", fontsize=10, color=TEXT_SECONDARY)
    axes.tick_params(axis="both", colors=TEXT_SECONDARY, length=0)
    axes.yaxis.grid(True, color=GRID, linewidth=1.0)
    axes.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axes.spines[spine].set_visible(False)
    axes.spines["bottom"].set_color(GRID)

    # Direct-label only the two extremes; the CSV carries every value.
    for position, align in ((0, "left"), (len(bins) - 1, "right")):
        bin_ = bins[position]
        offset = -width / 2.0 if align == "left" else width / 2.0
        axes.text(
            position + offset,
            bin_["total"] + max(totals) * 0.02,
            f"{bin_['faulty'] / bin_['total']:.0%} faulty",
            ha=align,
            va="bottom",
            fontsize=8.5,
            color=TEXT_SECONDARY,
            zorder=4,
        )

    figure.text(
        0.065,
        0.975,
        COHORT_TITLES[cohort],
        fontsize=14,
        color=TEXT_PRIMARY,
        va="top",
    )
    subtitle = (
        f"{summary['n_examples']:,} {unit} over "
        f"{summary['n_distinct_paths']:,} distinct paths "
        f"({DEFAULT_CELL_SIZE_M:.0f} m cells, {DEFAULT_HORIZON_S:.0f} s horizon); "
        f"{summary['n_singleton_paths']:,} of those paths occur exactly once. "
        f"{summary['n_faulty']:,} {unit} are faulty. Paths that leave the ego lane "
        f"cell hold {summary['lateral_examples'] / summary['n_examples']:.0%} of the "
        f"{unit} at a "
        f"{summary['lateral_faulty'] / summary['lateral_examples']:.0%} fault rate, "
        f"against {summary['straight_faulty'] / summary['straight_examples']:.0%} for "
        "the straight-corridor paths."
    )
    figure.text(
        0.065,
        0.925,
        textwrap.fill(subtitle, width=132),
        fontsize=9.5,
        color=TEXT_SECONDARY,
        va="top",
        linespacing=1.6,
    )
    axes.legend(
        handles=[
            Patch(facecolor=CLEAN_COLOR, label=COHORT_CLEAN_LABELS[cohort]),
            Patch(facecolor=FAULTY_COLOR, label=COHORT_FAULT_LABELS[cohort]),
        ],
        loc="lower left",
        bbox_to_anchor=(0.26, 1.02),
        ncols=2,
        frameon=False,
        fontsize=9.5,
        labelcolor=TEXT_SECONDARY,
        handlelength=1.4,
        handleheight=1.0,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def build_curve(records: list[dict]) -> tuple[dict, dict]:
    """Rank every path by example count and accumulate examples and faults."""
    total: Counter = Counter()
    faulty: Counter = Counter()
    for record in records:
        total[record["cells"]] += 1
        if record["faulty"]:
            faulty[record["cells"]] += 1

    ranked = sorted(total.items(), key=lambda item: (-item[1], item[0]))
    n_examples = len(records)
    n_faulty = sum(faulty.values())
    if not n_faulty:
        raise ValueError("Cohort has no faulty examples to distribute")

    counts, cumulative_examples, cumulative_faults = [], [], []
    running_examples = running_faults = 0
    for cells, count in ranked:
        running_examples += count
        running_faults += faulty[cells]
        counts.append(count)
        cumulative_examples.append(running_examples / n_examples)
        cumulative_faults.append(running_faults / n_faulty)

    # Fault density against how often a path recurs: the concentration question.
    buckets = (
        ("seen once", 1, 1),
        ("2-4x", 2, 4),
        ("5-19x", 5, 19),
        ("20x+", 20, None),
    )
    density = []
    for name, low, high in buckets:
        selected = [
            (cells, count)
            for cells, count in ranked
            if count >= low and (high is None or count <= high)
        ]
        examples = sum(count for _, count in selected)
        if examples:
            density.append(
                {
                    "bucket": name,
                    "n_paths": len(selected),
                    "examples": examples,
                    "faulty": sum(faulty[cells] for cells, _ in selected),
                }
            )

    curve = {
        "ranks": list(range(1, len(ranked) + 1)),
        "cells": [cells for cells, _ in ranked],
        "counts": counts,
        "faults": [faulty[cells] for cells, _ in ranked],
        "cumulative_examples": cumulative_examples,
        "cumulative_faults": cumulative_faults,
    }
    gaps = [
        example - fault
        for example, fault in zip(cumulative_examples, cumulative_faults, strict=True)
    ]
    peak = max(range(len(gaps)), key=lambda index: gaps[index])
    summary = {
        "n_examples": n_examples,
        "n_faulty": n_faulty,
        "n_paths": len(ranked),
        "n_singleton_paths": sum(1 for count in counts if count == 1),
        "max_gap_rank": peak + 1,
        "max_gap": gaps[peak],
        "density": density,
    }
    return curve, summary


def plot_curve(
    curve: dict,
    summary: dict,
    cohort: str,
    representation: str,
    output: Path,
) -> None:
    unit = COHORT_UNITS[cohort]
    if representation == "trajectory_grid":
        singular = "path"
        plural = "paths"
        title = COHORT_CURVE_TITLES[cohort]
    else:
        singular = "reasoning signature"
        plural = "reasoning signatures"
        title = (
            "Reasoning-coverage signature frequency and fault distribution: "
            "150 human-labeled cases"
        )
    ranks = curve["ranks"]
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(12.0, 8.4), sharex=True, height_ratios=(1.0, 1.0)
    )
    figure.patch.set_facecolor(SURFACE)
    axes_top = 0.75 if representation == "reasoning_path" else 0.80
    figure.subplots_adjust(
        left=0.085, right=0.86, top=axes_top, bottom=0.10, hspace=0.22
    )

    for axes in (top, bottom):
        axes.set_facecolor(SURFACE)
        axes.set_xscale("log")
        axes.set_xlim(1, ranks[-1] * 1.02)
        axes.grid(True, color=GRID, linewidth=1.0)
        axes.set_axisbelow(True)
        axes.tick_params(axis="both", colors=TEXT_SECONDARY, length=0)
        for spine in ("top", "right", "left"):
            axes.spines[spine].set_visible(False)
        axes.spines["bottom"].set_color(GRID)
    top.plot(ranks, curve["counts"], lw=2, color=CLEAN_COLOR, solid_joinstyle="round")
    top.plot(
        ranks,
        curve["faults"],
        lw=2,
        color=FAULTY_COLOR,
        marker="o",
        markersize=2.5,
        solid_joinstyle="round",
    )
    top.fill_between(
        ranks, curve["faults"], curve["counts"], color=CLEAN_COLOR, alpha=0.08, lw=0
    )
    top.set_ylim(-0.2, max(curve["counts"]) * 1.22)
    top.set_yticks(range(0, max(curve["counts"]) + 1, 2))
    top.set_ylabel(
        f"{unit.capitalize()} sharing the {singular}",
        fontsize=10,
        color=TEXT_SECONDARY,
    )
    top.set_title(
        f"Cases and faults per {singular}  "
        f"({summary['n_paths']:,} {plural}; log-ranked x-axis)",
        fontsize=10.5,
        color=TEXT_PRIMARY,
        loc="left",
        pad=8,
    )
    top.annotate(
        f"{curve['counts'][0]:,} {unit}",
        xy=(1, curve["counts"][0]),
        xytext=(1.35, curve["counts"][0] * 1.15),
        fontsize=8.5,
        color=TEXT_SECONDARY,
    )
    top.legend(
        handles=[
            Line2D([], [], color=CLEAN_COLOR, lw=2, label=f"All {unit}"),
            Line2D(
                [],
                [],
                color=FAULTY_COLOR,
                lw=2,
                marker="o",
                markersize=3,
                label="Faulty cases",
            ),
        ],
        loc="upper right",
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        handlelength=1.6,
    )
    first_singleton = next(
        (
            rank
            for rank, count in zip(ranks, curve["counts"], strict=True)
            if count == 1
        ),
        None,
    )
    if first_singleton:
        top.axvline(first_singleton, color=TEXT_SECONDARY, lw=1, ls=":", alpha=0.6)
        top.annotate(
            f"rank {first_singleton:,}: every\nremaining {singular} is unique",
            xy=(first_singleton, 1),
            xytext=(first_singleton * 1.12, max(curve["counts"]) * 0.34),
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )

    bottom.plot(
        ranks,
        curve["cumulative_examples"],
        lw=2,
        color=CLEAN_COLOR,
        solid_joinstyle="round",
    )
    bottom.plot(
        ranks,
        curve["cumulative_faults"],
        lw=2,
        color=FAULTY_COLOR,
        solid_joinstyle="round",
    )
    bottom.fill_between(
        ranks,
        curve["cumulative_faults"],
        curve["cumulative_examples"],
        color=FAULTY_COLOR,
        alpha=0.10,
        lw=0,
    )
    bottom.set_ylim(0, 1.04)
    bottom.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bottom.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    bottom.set_ylabel("Cumulative share", fontsize=10, color=TEXT_SECONDARY)
    bottom.set_xlabel(
        f"{singular.capitalize()} rank, ordered by {unit[:-1]} count (log scale)",
        fontsize=10,
        color=TEXT_SECONDARY,
        labelpad=8,
    )
    bottom.set_title(
        "Where the faults sit along that curve",
        fontsize=10.5,
        color=TEXT_PRIMARY,
        loc="left",
        pad=8,
    )
    bottom.legend(
        handles=[
            Line2D(
                [], [], color=CLEAN_COLOR, lw=2, label=f"Cumulative share of {unit}"
            ),
            Line2D(
                [], [], color=FAULTY_COLOR, lw=2, label="Cumulative share of faults"
            ),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        handlelength=1.6,
    )
    peak = summary["max_gap_rank"]
    # Anchor the callout to the panel's empty upper-left rather than to the peak,
    # which lands on the legend when the widest gap falls late in the ranking.
    bottom.annotate(
        f"widest gap at rank {peak:,}: the top {peak:,} {plural} hold\n"
        f"{curve['cumulative_examples'][peak - 1]:.0%} of {unit} but only "
        f"{curve['cumulative_faults'][peak - 1]:.0%} of faults",
        xy=(
            peak,
            (
                curve["cumulative_examples"][peak - 1]
                + curve["cumulative_faults"][peak - 1]
            )
            / 2,
        ),
        xytext=(0.03, 0.95),
        textcoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8.5,
        color=TEXT_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "lw": 0.8},
    )

    figure.text(
        0.085,
        0.975,
        title,
        fontsize=14,
        color=TEXT_PRIMARY,
        va="top",
    )
    density = " \u00b7 ".join(
        f"{row['bucket']}: {row['faulty'] / row['examples']:.0%}"
        for row in summary["density"]
    )
    if representation == "trajectory_grid":
        distribution_summary = "the rare paths in the tail are the fault-dense ones."
    else:
        distribution_summary = (
            "fault share largely tracks case share after the two most common "
            "reasoning signatures."
        )
    subtitle = (
        f"All {summary['n_paths']:,} {plural}, no bin folded away. "
        f"{summary['n_examples']:,} {unit}, {summary['n_faulty']:,} faulty "
        f"({summary['n_faulty'] / summary['n_examples']:.0%}). "
        f"Fault rate by how often a {singular} recurs \u2014 {density} \u2014 "
        f"{distribution_summary}"
    )
    figure.text(
        0.085,
        0.925,
        textwrap.fill(subtitle, width=118),
        fontsize=9.5,
        color=TEXT_SECONDARY,
        va="top",
        linespacing=1.6,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def _write_bins_table(bins: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "path", "cells", "n_examples", "n_faulty"])
        for rank, bin_ in enumerate(bins, start=1):
            writer.writerow(
                [
                    rank,
                    bin_["label"].replace("\n", " "),
                    (
                        ";".join(f"{x},{y}" for x, y in bin_["cells"])
                        if bin_["cells"]
                        else ""
                    ),
                    bin_["total"],
                    bin_["faulty"],
                ]
            )


def _reasoning_signature_label(items: tuple[str, ...]) -> str:
    typed_paths = [
        item.removeprefix("table:path=")
        for item in items
        if item.startswith("table:path=")
    ]
    if typed_paths:
        return " & ".join(typed_paths)
    decision = next(
        (
            item.removeprefix("table:z3_pair=")
            for item in items
            if item.startswith("table:z3_pair=")
        ),
        "none|none",
    )
    return f"unobserved>{decision}"


def _write_curve_table(curve: dict, path: Path, representation: str) -> None:
    """Write every ranked coverage signature underlying a curve."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        signature_name = (
            "path" if representation == "trajectory_grid" else "reasoning_signature"
        )
        items_name = (
            "cells" if representation == "trajectory_grid" else "coverage_items"
        )
        writer.writerow(
            [
                "rank",
                signature_name,
                items_name,
                "n_examples",
                "n_faulty",
                "cumulative_example_share",
                "cumulative_fault_share",
            ]
        )
        rows = zip(
            curve["ranks"],
            curve["cells"],
            curve["counts"],
            curve["faults"],
            curve["cumulative_examples"],
            curve["cumulative_faults"],
            strict=True,
        )
        for rank, items, count, faults, example_share, fault_share in rows:
            if representation == "trajectory_grid":
                signature = _path_label(items)
                serialized_items = ";".join(f"{x},{y}" for x, y in items)
            else:
                signature = _reasoning_signature_label(items)
                serialized_items = ";".join(items)
            writer.writerow(
                [
                    rank,
                    signature,
                    serialized_items,
                    count,
                    faults,
                    f"{example_share:.6f}",
                    f"{fault_share:.6f}",
                ]
            )


CELL_FREQUENCY_BUCKETS = (
    ("crossed once", 1, 1),
    ("2-9x", 2, 9),
    ("10-99x", 10, 99),
    ("100x+", 100, None),
)


def build_cells(records: list[dict]) -> tuple[dict, dict]:
    """Rank every grid cell by how many examples cross it, and accumulate faults.

    Each example contributes one count to every cell its plan crosses, so a plan
    covering five cells adds one to each of those five coverage counts and, when
    the example is faulty, one to each of the same five fault counts. Counts are
    therefore cell *crossings*, not examples: they sum to more than the cohort.
    """
    coverage: Counter = Counter()
    faults: Counter = Counter()
    for record in records:
        for cell in record["cells"]:
            coverage[cell] += 1
            if record["faulty"]:
                faults[cell] += 1
    if not coverage:
        raise ValueError("No covered cells to plot")

    ranked = sorted(coverage.items(), key=lambda item: (-item[1], item[0]))
    total_crossings = sum(coverage.values())
    total_fault_crossings = sum(faults.values())
    if not total_fault_crossings:
        raise ValueError("Cohort has no faulty crossings to distribute")

    counts, cumulative_crossings, cumulative_faults = [], [], []
    running_crossings = running_faults = 0
    for cell, count in ranked:
        running_crossings += count
        running_faults += faults[cell]
        counts.append(count)
        cumulative_crossings.append(running_crossings / total_crossings)
        cumulative_faults.append(running_faults / total_fault_crossings)

    density = []
    for name, low, high in CELL_FREQUENCY_BUCKETS:
        selected = [
            cell
            for cell, count in ranked
            if count >= low and (high is None or count <= high)
        ]
        crossings = sum(coverage[cell] for cell in selected)
        if crossings:
            density.append(
                {
                    "bucket": name,
                    "n_cells": len(selected),
                    "crossings": crossings,
                    "faulty": sum(faults[cell] for cell in selected),
                }
            )

    curve = {
        "ranks": list(range(1, len(ranked) + 1)),
        "cells": [cell for cell, _ in ranked],
        "counts": counts,
        "faults": [faults[cell] for cell, _ in ranked],
        "cumulative_crossings": cumulative_crossings,
        "cumulative_faults": cumulative_faults,
    }
    gaps = [
        crossing - fault
        for crossing, fault in zip(cumulative_crossings, cumulative_faults, strict=True)
    ]
    peak = max(range(len(gaps)), key=lambda index: gaps[index])
    centerline_mass = sum(count for cell, count in coverage.items() if cell[1] == 0)
    summary = {
        "n_examples": len(records),
        "n_faulty": sum(1 for record in records if record["faulty"]),
        "n_cells": len(ranked),
        "n_singleton_cells": sum(1 for count in counts if count == 1),
        "total_crossings": total_crossings,
        "total_fault_crossings": total_fault_crossings,
        "forward_reach_m": max(cell[0] for cell in coverage) * DEFAULT_CELL_SIZE_M,
        "lateral_span_m": (
            min(cell[1] for cell in coverage) * DEFAULT_CELL_SIZE_M,
            max(cell[1] for cell in coverage) * DEFAULT_CELL_SIZE_M,
        ),
        "centerline_share_of_coverage": centerline_mass / total_crossings,
        "max_gap_rank": peak + 1,
        "max_gap": gaps[peak],
        "density": density,
    }
    return curve, summary


def _cell_label(cell: tuple[int, int]) -> str:
    forward = cell[0] * DEFAULT_CELL_SIZE_M
    lateral = cell[1] * DEFAULT_CELL_SIZE_M
    if not lateral:
        return f"{forward:.0f} m ahead, centerline"
    side = "left" if lateral > 0 else "right"
    return f"{forward:.0f} m ahead, {abs(lateral):.0f} m {side}"


def plot_cells(curve: dict, summary: dict, cohort: str, output: Path) -> None:
    unit = COHORT_UNITS[cohort]
    ranks = curve["ranks"]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(12.0, 8.4), sharex=True)
    figure.patch.set_facecolor(SURFACE)
    figure.subplots_adjust(left=0.085, right=0.86, top=0.80, bottom=0.10, hspace=0.22)

    for axes in (top, bottom):
        axes.set_facecolor(SURFACE)
        axes.set_xscale("log")
        axes.set_xlim(1, ranks[-1] * 1.02)
        axes.grid(True, color=GRID, linewidth=1.0)
        axes.set_axisbelow(True)
        axes.tick_params(axis="both", colors=TEXT_SECONDARY, length=0)
        for spine in ("top", "right", "left"):
            axes.spines[spine].set_visible(False)
        axes.spines["bottom"].set_color(GRID)

    floor = 0.9
    top.plot(ranks, curve["counts"], lw=2, color=CLEAN_COLOR, solid_joinstyle="round")
    top.fill_between(ranks, curve["counts"], floor, color=CLEAN_COLOR, alpha=0.10, lw=0)
    # A log axis cannot show a zero, so break the fault line where a cell has no
    # faulty crossing and rug those cells on the baseline instead of dropping them.
    top.plot(
        ranks,
        [faulty if faulty else float("nan") for faulty in curve["faults"]],
        lw=2,
        color=FAULTY_COLOR,
        solid_joinstyle="round",
    )
    zero_fault_ranks = [
        rank for rank, faulty in zip(ranks, curve["faults"], strict=True) if not faulty
    ]
    if zero_fault_ranks:
        top.plot(
            zero_fault_ranks,
            [floor] * len(zero_fault_ranks),
            linestyle="none",
            marker="|",
            markersize=5,
            color=FAULTY_COLOR,
            alpha=0.55,
        )
    top.set_yscale("log")
    top.set_ylim(floor, max(curve["counts"]) * 1.6)
    top.set_ylabel(
        f"{unit.capitalize()} crossing the cell", fontsize=10, color=TEXT_SECONDARY
    )
    top.set_title(
        f"How steeply cell coverage falls off  ({summary['n_cells']:,} cells, log-log)",
        fontsize=10.5,
        color=TEXT_PRIMARY,
        loc="left",
        pad=8,
    )
    top.legend(
        handles=[
            Line2D(
                [],
                [],
                color=CLEAN_COLOR,
                lw=2,
                label=f"{unit.capitalize()} crossing the cell",
            ),
            Line2D(
                [],
                [],
                color=FAULTY_COLOR,
                lw=2,
                label=(
                    "Faulty ones among them"
                    + (
                        f"  (tick = 0, {len(zero_fault_ranks)} cells)"
                        if zero_fault_ranks
                        else ""
                    )
                ),
            ),
        ],
        loc="upper right",
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        handlelength=1.6,
    )
    top.annotate(
        f"{_cell_label(curve['cells'][0])}: every {unit[:-1]} crosses it",
        xy=(1, curve["counts"][0]),
        xytext=(1.35, curve["counts"][0] * 1.1),
        fontsize=8.5,
        color=TEXT_SECONDARY,
    )
    first_singleton = next(
        (
            rank
            for rank, count in zip(ranks, curve["counts"], strict=True)
            if count == 1
        ),
        None,
    )
    if first_singleton:
        top.axvline(first_singleton, color=TEXT_SECONDARY, lw=1, ls=":", alpha=0.6)
        top.annotate(
            f"rank {first_singleton:,}: cells crossed\nby a single {unit[:-1]}",
            xy=(first_singleton, 1),
            xytext=(first_singleton * 1.1, 2.4),
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )

    bottom.plot(
        ranks,
        curve["cumulative_crossings"],
        lw=2,
        color=CLEAN_COLOR,
        solid_joinstyle="round",
    )
    bottom.plot(
        ranks,
        curve["cumulative_faults"],
        lw=2,
        color=FAULTY_COLOR,
        solid_joinstyle="round",
    )
    bottom.fill_between(
        ranks,
        curve["cumulative_faults"],
        curve["cumulative_crossings"],
        color=FAULTY_COLOR,
        alpha=0.10,
        lw=0,
    )
    bottom.set_ylim(0, 1.04)
    bottom.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bottom.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    bottom.set_ylabel("Cumulative share", fontsize=10, color=TEXT_SECONDARY)
    bottom.set_xlabel(
        "Cell rank, ordered by how many examples cross it (log scale)",
        fontsize=10,
        color=TEXT_SECONDARY,
        labelpad=8,
    )
    bottom.set_title(
        "Where the faulty crossings sit along that curve",
        fontsize=10.5,
        color=TEXT_PRIMARY,
        loc="left",
        pad=8,
    )
    bottom.legend(
        handles=[
            Line2D(
                [],
                [],
                color=CLEAN_COLOR,
                lw=2,
                label="Cumulative share of cell crossings",
            ),
            Line2D(
                [],
                [],
                color=FAULTY_COLOR,
                lw=2,
                label="Cumulative share of faulty crossings",
            ),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        handlelength=1.6,
    )
    peak = summary["max_gap_rank"]
    bottom.annotate(
        f"widest gap at rank {peak:,}: the top {peak:,} cells hold\n"
        f"{curve['cumulative_crossings'][peak - 1]:.0%} of crossings but only "
        f"{curve['cumulative_faults'][peak - 1]:.0%} of faulty ones",
        xy=(
            peak,
            (
                curve["cumulative_crossings"][peak - 1]
                + curve["cumulative_faults"][peak - 1]
            )
            / 2,
        ),
        xytext=(0.03, 0.95),
        textcoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8.5,
        color=TEXT_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "lw": 0.8},
    )

    figure.text(
        0.085,
        0.975,
        COHORT_CELL_TITLES[cohort],
        fontsize=14,
        color=TEXT_PRIMARY,
        va="top",
    )
    density = " \u00b7 ".join(
        f"{row['bucket']}: {row['faulty'] / row['crossings']:.0%}"
        for row in summary["density"]
    )
    low, high = summary["lateral_span_m"]
    subtitle = (
        f"Every {unit[:-1]} adds one count to each cell its 3 s plan crosses, and a "
        f"faulty one adds to the fault count of those same cells. "
        f"{summary['n_examples']:,} {unit} produce "
        f"{summary['total_crossings']:,} crossings over {summary['n_cells']:,} cells, "
        f"reaching {summary['forward_reach_m']:.0f} m ahead and {low:.0f} to "
        f"{high:.0f} m laterally, though "
        f"{summary['centerline_share_of_coverage']:.0%} of crossings land on the "
        f"centerline. Fault rate by how often a cell is crossed \u2014 {density}."
    )
    figure.text(
        0.085,
        0.925,
        textwrap.fill(subtitle, width=118),
        fontsize=9.5,
        color=TEXT_SECONDARY,
        va="top",
        linespacing=1.6,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def _write_cells_table(curve: dict, path: Path) -> None:
    """Every cell, so any rank on the curve can be located on request."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "rank",
                "cell",
                "location",
                "forward_m",
                "lateral_m",
                "n_examples",
                "n_faulty",
                "fault_rate",
                "cumulative_crossing_share",
                "cumulative_fault_share",
            ]
        )
        rows = zip(
            curve["ranks"],
            curve["cells"],
            curve["counts"],
            curve["faults"],
            curve["cumulative_crossings"],
            curve["cumulative_faults"],
            strict=True,
        )
        for rank, cell, count, faulty, crossing_share, fault_share in rows:
            writer.writerow(
                [
                    rank,
                    f"{cell[0]},{cell[1]}",
                    _cell_label(cell),
                    f"{cell[0] * DEFAULT_CELL_SIZE_M:.0f}",
                    f"{cell[1] * DEFAULT_CELL_SIZE_M:.0f}",
                    count,
                    faulty,
                    f"{faulty / count:.6f}",
                    f"{crossing_share:.6f}",
                    f"{fault_share:.6f}",
                ]
            )


def main() -> int:
    args = parse_args()
    if args.top_paths <= 0:
        raise ValueError("--top-paths must be positive")
    records = collect_paths(args)
    table_path = args.output.with_suffix(".csv")

    if args.layout == "cells":
        cells, summary = build_cells(records)
        plot_cells(cells, summary, args.cohort, args.output)
        _write_cells_table(cells, table_path)
    elif args.layout == "curve":
        curve, summary = build_curve(records)
        plot_curve(
            curve,
            summary,
            args.cohort,
            args.representation,
            args.output,
        )
        _write_curve_table(curve, table_path, args.representation)
    else:
        bins, summary = build_bins(records, args.top_paths)
        plot(bins, summary, args.cohort, args.output)
        _write_bins_table(bins, table_path)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output} and {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
