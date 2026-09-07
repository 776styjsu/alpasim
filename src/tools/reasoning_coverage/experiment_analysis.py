# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Plot and characterize the Alpamayo reasoning-coverage experiments.

This module deliberately keeps fault labels separate from coverage construction.
Labels are joined only after every outcome-blind prioritization order has been
constructed.  Reported intervals therefore describe randomized greedy
tie-breaking, not independent-dataset uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import numpy as np
from trajectory_safety.planned_center_labels import COLLISION_OUTCOME, OFFROAD_OUTCOME

from .analysis import SCENE_FIELDS, apfd, build_cases
from .analysis import build_coverage_sets as build_benchmark_coverage
from .ood_evaluation import (
    _development_scene_ids,
    _load_or_extract_frames,
    lazy_greedy_coverage_order,
)
from .program import lexical_items
from .table_program import (
    TableReasoningProgram,
    atomic_reasoning_path_items,
    compile_table_program,
    primary_reasoning_items,
    sequence_reasoning_items,
)
from .trajectory_grid import (
    DEFAULT_CELL_SIZE_M,
    DEFAULT_HORIZON_S,
    trajectory_grid_cell,
    trajectory_grid_items,
)

DISPLAY_NAMES = {
    "random": "Random",
    "specification": "NuRec scene spec",
    "visual_stats": "Visual summary",
    "image_grid": "Image grid",
    "trajectory": "Trajectory summary",
    "trajectory_grid": "Trajectory grid",
    "reasoning_path": "Full reasoning Z1/Z2/Z3",
    "atomic_reasoning_path": "Atomic reasoning Z1/Z2/Z3",
    "reasoning_continuation": "Reasoning + continuation",
    "text_bigram": "Text bigram",
}
COLORS = {
    "random": "#7f7f7f",
    "specification": "#9467bd",
    "visual_stats": "#8c564b",
    "image_grid": "#e377c2",
    "trajectory": "#17becf",
    "trajectory_grid": "#ff7f0e",
    "reasoning_path": "#1f77b4",
    "atomic_reasoning_path": "#0b4f8a",
    "reasoning_continuation": "#2ca02c",
    "text_bigram": "#d62728",
}
BENCHMARK_OUTCOMES = (
    "inconsistent_reliable_cot",
    "unsafe_trajectory",
)
PLANNED_OUTCOMES = (COLLISION_OUTCOME, OFFROAD_OUTCOME)
OUTCOME_NAMES = {
    "inconsistent_reliable_cot": "Inconsistent reliable CoT",
    "unsafe_trajectory": "Unsafe trajectory",
    COLLISION_OUTCOME: "Collision-positive frame",
    OFFROAD_OUTCOME: "Center-off-road-positive frame",
}
ENTITY_PREDICATES = {
    "lead_slow",
    "lead_stopped",
    "merging_or_cutin_agent",
    "crossing_agent",
    "oncoming_agent",
    "adjacent_agent",
    "vru_present",
    "static_obstacle",
}
ROAD_CONTEXT_BY_PREDICATE = {
    "intersection_or_turn": "intersection or turn",
    "road_curve": "curve",
    "merge_split_or_ramp": "merge, split, or ramp",
    "workzone_or_road_event": "work zone or road event",
    "speed_constraint": "speed-constrained road",
}
BENCHMARK_BUDGETS = (0.05, 0.1, 0.25, 0.5)
PLANNED_BUDGETS = (0.01, 0.05, 0.1, 0.25)
ADVANTAGE_BUDGETS = (0.1, 0.25)
BENCHMARK_DISCOVERY_BUDGETS = (0.05, 0.1, 0.25)
PLANNED_DISCOVERY_BUDGETS = (0.01, 0.05, 0.1, 0.25)
REASONING_CANDIDATES = ("reasoning_path", "atomic_reasoning_path")
SPATIAL_GRID_BUDGET = 0.1


@dataclass(frozen=True)
class DatasetAnalysis:
    """Outcome-blind orders and fault units for one evaluated cohort."""

    case_ids: tuple[str, ...]
    coverage: dict[str, dict[str, frozenset[str]]]
    orders: dict[str, list[list[str]]]
    outcomes: dict[str, dict[str, frozenset[str]]]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = next(
            (
                payload[key]
                for key in ("entries", "results", "frames")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
    else:
        values = None
    if not isinstance(values, list) or not all(
        isinstance(item, dict) for item in values
    ):
        raise ValueError("Expected a JSON list of objects")
    return values


def _orders(
    case_ids: Iterable[str],
    coverage: dict[str, dict[str, frozenset[str]]],
    *,
    repeats: int,
    seed: int,
    multi_round: bool,
) -> dict[str, list[list[str]]]:
    ordered_ids = sorted(case_ids)
    result = {arm: [] for arm in ("random", *coverage)}
    for repeat in range(repeats):
        repeat_seed = seed + repeat
        random_order = list(ordered_ids)
        random.Random(repeat_seed).shuffle(random_order)
        result["random"].append(random_order)
        for arm, items in coverage.items():
            result[arm].append(
                lazy_greedy_coverage_order(
                    ordered_ids, items, repeat_seed, multi_round=multi_round
                )
            )
    return result


def _single_case_faults(case_ids: Iterable[str]) -> dict[str, frozenset[str]]:
    return {case_id: frozenset((case_id,)) for case_id in sorted(case_ids)}


def _descending_score_orders(
    case_ids: Iterable[str],
    scores: dict[str, float | int],
    *,
    repeats: int,
    seed: int,
) -> list[list[str]]:
    """Order by descending score with the experiment's randomized tie breaks."""
    ordered_ids = sorted(case_ids)
    result = []
    for repeat in range(repeats):
        order = list(ordered_ids)
        random.Random(seed + repeat).shuffle(order)
        order.sort(key=lambda case_id: -scores[case_id])
        result.append(order)
    return result


def _mean_apfd(orders: list[list[str]], frame_failures: set[str]) -> float:
    return fmean(float(apfd(order, frame_failures)) for order in orders)


def _planned_text_bigram_diagnostic(
    analysis: DatasetAnalysis,
    frame_text: dict[str, str],
    frame_failures: dict[str, set[str]],
    *,
    repeats: int,
    seed: int,
    multi_round: bool,
) -> dict[str, Any]:
    """Diagnose why raw text coverage prioritizes center-off-road outcomes."""
    item_sets = analysis.coverage["text_bigram"]
    item_support = Counter(
        item for case_id in analysis.case_ids for item in item_sets[case_id]
    )
    support2_sets = {
        case_id: frozenset(item for item in items if item_support[item] >= 2)
        for case_id, items in item_sets.items()
    }
    support2_orders = _orders(
        analysis.case_ids,
        {"support2": support2_sets},
        repeats=repeats,
        seed=seed,
        multi_round=multi_round,
    )["support2"]
    word_count = {
        case_id: len(re.findall(r"[a-z0-9]+", frame_text[case_id].lower()))
        for case_id in analysis.case_ids
    }
    inverse_support = {
        case_id: sum(1.0 / item_support[item] for item in item_sets[case_id])
        for case_id in analysis.case_ids
    }
    diagnostic_orders = {
        "random": analysis.orders["random"],
        "text_bigram": analysis.orders["text_bigram"],
        "text_bigram_support2": support2_orders,
        "cot_length": _descending_score_orders(
            analysis.case_ids, word_count, repeats=repeats, seed=seed
        ),
        "inverse_bigram_support": _descending_score_orders(
            analysis.case_ids, inverse_support, repeats=repeats, seed=seed
        ),
    }
    apfd_results = {
        outcome: {
            arm: _mean_apfd(orders, frame_failures[outcome])
            for arm, orders in diagnostic_orders.items()
        }
        for outcome in PLANNED_OUTCOMES
    }

    offroad = frame_failures[OFFROAD_OUTCOME]
    non_offroad = set(analysis.case_ids) - offroad
    feature_values = {
        "cot_words": word_count,
        "unique_bigrams": {
            case_id: len(item_sets[case_id]) for case_id in analysis.case_ids
        },
        "support_le_2_bigrams": {
            case_id: sum(item_support[item] <= 2 for item in item_sets[case_id])
            for case_id in analysis.case_ids
        },
        "inverse_bigram_support": inverse_support,
    }
    feature_summary = {
        feature: {
            "offroad_mean": fmean(values[case_id] for case_id in offroad),
            "non_offroad_mean": fmean(values[case_id] for case_id in non_offroad),
        }
        for feature, values in feature_values.items()
    }

    positive_support = Counter(
        item for case_id in offroad for item in item_sets[case_id]
    )
    prevalence = len(offroad) / len(analysis.case_ids)
    phrase_rows = []
    for item, support in item_support.items():
        positive = positive_support[item]
        if support < 50 or positive < 5:
            continue
        phrase_rows.append(
            {
                "bigram": item.removeprefix("text:bigram=").replace("_", " "),
                "support": support,
                "offroad_frames": positive,
                "offroad_fraction": positive / support,
                "excess_offroad_frames": positive - support * prevalence,
            }
        )
    phrase_rows.sort(
        key=lambda row: (row["excess_offroad_frames"], row["offroad_fraction"]),
        reverse=True,
    )

    coverage_universe = frozenset(item_support)
    saturation_rows = []
    for order in analysis.orders["text_bigram"]:
        covered: set[str] = set()
        for rank, case_id in enumerate(order, start=1):
            covered.update(item_sets[case_id])
            if len(covered) == len(coverage_universe):
                selected = set(order[:rank])
                saturation_rows.append(
                    {
                        "rank": rank,
                        "offroad_frames": len(selected & offroad),
                        "offroad_fraction": len(selected & offroad) / rank,
                    }
                )
                break
    return {
        "n_cases": len(analysis.case_ids),
        "n_offroad_frames": len(offroad),
        "offroad_prevalence": prevalence,
        "n_text_bigrams": len(item_support),
        "n_support_1_bigrams": sum(value == 1 for value in item_support.values()),
        "n_support_le_2_bigrams": sum(value <= 2 for value in item_support.values()),
        "feature_summary": feature_summary,
        "apfd": apfd_results,
        "top_offroad_associated_bigrams": phrase_rows[:8],
        "raw_bigram_saturation": {
            "median_cases": float(np.median([row["rank"] for row in saturation_rows])),
            "mean_offroad_frames": fmean(
                row["offroad_frames"] for row in saturation_rows
            ),
            "mean_offroad_fraction": fmean(
                row["offroad_fraction"] for row in saturation_rows
            ),
        },
    }


def _detection_ranks(
    order: list[str], fault_units: dict[str, frozenset[str]]
) -> dict[str, int]:
    rank = {case_id: index for index, case_id in enumerate(order, start=1)}
    return {
        fault_id: min(rank[case_id] for case_id in cases)
        for fault_id, cases in fault_units.items()
    }


def _budget_count(n_cases: int, fraction: float) -> int:
    if fraction <= 0.0:
        return 0
    return min(n_cases, max(1, math.ceil(n_cases * fraction)))


def _curve_fractions(n_cases: int, points: int) -> tuple[float, ...]:
    if n_cases <= points:
        return tuple(index / n_cases for index in range(n_cases + 1))
    return tuple(float(value) for value in np.linspace(0.0, 1.0, points + 1))


def _curve_summary(
    arm_orders: list[list[str]],
    fault_units: dict[str, frozenset[str]],
    fractions: tuple[float, ...],
) -> dict[str, list[float] | list[int]]:
    n_cases = len(arm_orders[0])
    budgets = [_budget_count(n_cases, fraction) for fraction in fractions]
    values: list[list[int]] = []
    for order in arm_orders:
        ranks = np.asarray(list(_detection_ranks(order, fault_units).values()))
        values.append([int((ranks <= budget).sum()) for budget in budgets])
    array = np.asarray(values, dtype=float)
    return {
        "budget_fraction": list(fractions),
        "budget_cases": budgets,
        "mean_faults": array.mean(axis=0).tolist(),
        "p2_5_faults": np.quantile(array, 0.025, axis=0).tolist(),
        "p97_5_faults": np.quantile(array, 0.975, axis=0).tolist(),
    }


def _coverage_curve_summary(
    arm_orders: list[list[str]],
    item_sets: dict[str, frozenset[str]],
    fractions: tuple[float, ...],
) -> dict[str, Any]:
    """Summarize observed-item coverage and its first saturation point."""
    n_cases = len(arm_orders[0])
    budgets = [_budget_count(n_cases, fraction) for fraction in fractions]
    universe = frozenset().union(*item_sets.values())
    if not universe:
        raise ValueError("Coverage arm has an empty observed-item universe")

    values: list[list[float]] = []
    saturation_cases: list[int] = []
    for order in arm_orders:
        covered: set[str] = set()
        coverage_by_budget = [0.0]
        saturation = None
        for index, case_id in enumerate(order, start=1):
            covered.update(item_sets[case_id])
            coverage_by_budget.append(len(covered) / len(universe))
            if saturation is None and len(covered) == len(universe):
                saturation = index
        if saturation is None:
            raise ValueError("Observed coverage did not saturate after all cases")
        values.append([coverage_by_budget[budget] for budget in budgets])
        saturation_cases.append(saturation)

    array = np.asarray(values, dtype=float)
    saturation_array = np.asarray(saturation_cases, dtype=float)
    return {
        "n_observed_items": len(universe),
        "budget_fraction": list(fractions),
        "budget_cases": budgets,
        "mean_coverage": array.mean(axis=0).tolist(),
        "p2_5_coverage": np.quantile(array, 0.025, axis=0).tolist(),
        "p97_5_coverage": np.quantile(array, 0.975, axis=0).tolist(),
        "saturation_cases": {
            "median": float(np.median(saturation_array)),
            "p2_5": float(np.quantile(saturation_array, 0.025)),
            "p97_5": float(np.quantile(saturation_array, 0.975)),
            "min": int(saturation_array.min()),
            "max": int(saturation_array.max()),
        },
    }


def _coverage_report(
    analysis: DatasetAnalysis, *, points: int
) -> dict[str, dict[str, Any]]:
    fractions = _curve_fractions(len(analysis.case_ids), points)
    return {
        arm: _coverage_curve_summary(analysis.orders[arm], item_sets, fractions)
        for arm, item_sets in analysis.coverage.items()
    }


def _probabilities_at_budget(
    arm_orders: list[list[str]],
    fault_units: dict[str, frozenset[str]],
    fraction: float,
) -> dict[str, float]:
    budget = _budget_count(len(arm_orders[0]), fraction)
    selected = Counter()
    for order in arm_orders:
        for fault_id, rank in _detection_ranks(order, fault_units).items():
            selected[fault_id] += rank <= budget
    return {fault_id: selected[fault_id] / len(arm_orders) for fault_id in fault_units}


def _distribution(values: list[float | int]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p2_5": float(np.quantile(array, 0.025)),
        "p97_5": float(np.quantile(array, 0.975)),
    }


def _fault_unit_apfd(order: list[str], fault_units: dict[str, frozenset[str]]) -> float:
    ranks = _detection_ranks(order, fault_units)
    return (
        1.0 - sum(ranks.values()) / (len(order) * len(ranks)) + 1.0 / (2.0 * len(order))
    )


def _detected_faults(
    order: list[str],
    fault_units: dict[str, frozenset[str]],
    budget: int,
) -> set[str]:
    selected = set(order[:budget])
    return {
        fault_id for fault_id, cases in fault_units.items() if selected & set(cases)
    }


def _exclusive_summary(
    detected_by_arm: dict[str, list[set[str]]],
    fault_ids: set[str],
    *,
    candidate_arms: tuple[str, ...],
    comparator_arms: tuple[str, ...],
    limit: int = 50,
) -> tuple[dict[str, Any], dict[str, float]]:
    if not candidate_arms or not comparator_arms:
        raise ValueError("Exclusive discovery requires candidates and comparators")
    repeat_counts = {
        len(detected_by_arm[arm]) for arm in (*candidate_arms, *comparator_arms)
    }
    if len(repeat_counts) != 1:
        raise ValueError("Compared arms have different repeat counts")
    repeats = repeat_counts.pop()
    exclusive_counts = Counter()
    candidate_counts = Counter()
    comparator_counts = Counter()
    overlap_counts = Counter()
    count_rows = []
    for repeat in range(repeats):
        candidate = set().union(
            *(detected_by_arm[arm][repeat] for arm in candidate_arms)
        )
        comparator = set().union(
            *(detected_by_arm[arm][repeat] for arm in comparator_arms)
        )
        exclusive = candidate - comparator
        candidate_only = len(exclusive)
        comparator_only = len(comparator - candidate)
        overlap = len(candidate & comparator)
        count_rows.append((candidate_only, comparator_only, overlap))
        candidate_counts.update(candidate)
        comparator_counts.update(comparator)
        overlap_counts.update(candidate & comparator)
        exclusive_counts.update(exclusive)

    candidate_probability = {
        fault_id: candidate_counts[fault_id] / repeats for fault_id in fault_ids
    }
    comparator_probability = {
        fault_id: comparator_counts[fault_id] / repeats for fault_id in fault_ids
    }
    exclusive_probability = {
        fault_id: exclusive_counts[fault_id] / repeats for fault_id in fault_ids
    }
    ranked = sorted(
        fault_ids,
        key=lambda fault_id: (
            -exclusive_probability[fault_id],
            -candidate_probability[fault_id],
            fault_id,
        ),
    )
    robust = [
        fault_id
        for fault_id in ranked
        if candidate_probability[fault_id] >= 0.75
        and comparator_probability[fault_id] <= 0.25
    ]
    summary = {
        "candidate_arms": list(candidate_arms),
        "comparator_arms": list(comparator_arms),
        "candidate_only_faults": _distribution([row[0] for row in count_rows]),
        "comparator_only_faults": _distribution([row[1] for row in count_rows]),
        "overlapping_faults": _distribution([row[2] for row in count_rows]),
        "n_faults_with_nonzero_candidate_only_probability": sum(
            probability > 0.0 for probability in exclusive_probability.values()
        ),
        "n_faults_with_candidate_only_probability_ge_0_5": sum(
            probability >= 0.5 for probability in exclusive_probability.values()
        ),
        "robustly_exclusive_definition": (
            "candidate-union detection probability >= 0.75 and comparator-union "
            "detection probability <= 0.25"
        ),
        "robustly_exclusive_faults": robust,
        "top_candidate_only_faults": [
            {
                "fault_id": fault_id,
                "candidate_detection_probability": candidate_probability[fault_id],
                "comparator_union_detection_probability": comparator_probability[
                    fault_id
                ],
                "candidate_only_probability": exclusive_probability[fault_id],
            }
            for fault_id in ranked[:limit]
            if exclusive_probability[fault_id] > 0.0
        ],
    }
    return summary, exclusive_probability


def _early_fault_discovery_report(
    analysis: DatasetAnalysis,
    outcomes: Iterable[str],
    budgets: tuple[float, ...],
    non_reasoning_arms: tuple[str, ...],
    context_features: dict[str, dict[str, set[str]]],
    *,
    fault_unit: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing = set((*REASONING_CANDIDATES, *non_reasoning_arms)) - set(analysis.orders)
    if missing:
        raise ValueError(f"Early-discovery arms are unavailable: {sorted(missing)}")
    nonrandom_arms = tuple(arm for arm in analysis.orders if arm != "random")
    report: dict[str, Any] = {
        "fault_unit": fault_unit,
        "probability_semantics": (
            "Fraction of randomized greedy tie-break repeats in which the fault "
            "is detected within the stated budget"
        ),
        "paired_exclusivity_semantics": (
            "Candidate-only counts compare same-seed repeats. Against a comparator "
            "union, each comparator receives the full stated budget, so this is a "
            "strict overlap screen rather than a budget-matched portfolio."
        ),
        "reasoning_candidates": list(REASONING_CANDIDATES),
        "non_reasoning_comparators": list(non_reasoning_arms),
        "outcomes": {},
    }
    csv_rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        fault_units = analysis.outcomes[outcome]
        fault_ids = set(fault_units)
        outcome_report: dict[str, Any] = {
            "n_faults": len(fault_ids),
            "apfd": {
                arm: _distribution(
                    [_fault_unit_apfd(order, fault_units) for order in orders]
                )
                for arm, orders in analysis.orders.items()
            },
            "budgets": {},
        }
        for fraction in budgets:
            budget = _budget_count(len(analysis.case_ids), fraction)
            detected_by_arm = {
                arm: [_detected_faults(order, fault_units, budget) for order in orders]
                for arm, orders in analysis.orders.items()
            }
            arm_probabilities = {
                arm: {
                    fault_id: sum(fault_id in detected for detected in detected_repeats)
                    / len(detected_repeats)
                    for fault_id in fault_ids
                }
                for arm, detected_repeats in detected_by_arm.items()
            }
            arm_detection = {}
            for arm, detected_repeats in detected_by_arm.items():
                probabilities = arm_probabilities[arm]
                arm_detection[arm] = {
                    "detected_faults": _distribution(
                        [len(detected) for detected in detected_repeats]
                    ),
                    "mean_recall": fmean(probabilities.values()),
                    "n_faults_detection_probability_ge_0_75": sum(
                        value >= 0.75 for value in probabilities.values()
                    ),
                }
            candidate_report = {}
            exclusive_probabilities = {}
            reasoning_pairwise_probabilities = {}
            for candidate in REASONING_CANDIDATES:
                pairwise = {}
                for comparator in non_reasoning_arms:
                    pairwise[comparator], _ = _exclusive_summary(
                        detected_by_arm,
                        fault_ids,
                        candidate_arms=(candidate,),
                        comparator_arms=(comparator,),
                    )
                pairwise_other_coverage = {}
                for comparator in nonrandom_arms:
                    if comparator == candidate:
                        continue
                    (
                        pairwise_other_coverage[comparator],
                        pairwise_probabilities,
                    ) = _exclusive_summary(
                        detected_by_arm,
                        fault_ids,
                        candidate_arms=(candidate,),
                        comparator_arms=(comparator,),
                    )
                    if comparator in REASONING_CANDIDATES:
                        reasoning_pairwise_probabilities[(candidate, comparator)] = (
                            pairwise_probabilities
                        )
                versus_non_reasoning, exclusive = _exclusive_summary(
                    detected_by_arm,
                    fault_ids,
                    candidate_arms=(candidate,),
                    comparator_arms=non_reasoning_arms,
                )
                versus_all, _ = _exclusive_summary(
                    detected_by_arm,
                    fault_ids,
                    candidate_arms=(candidate,),
                    comparator_arms=tuple(
                        arm for arm in nonrandom_arms if arm != candidate
                    ),
                )
                candidate_report[candidate] = {
                    "pairwise": pairwise,
                    "pairwise_other_coverage": pairwise_other_coverage,
                    "versus_non_reasoning_union": versus_non_reasoning,
                    "versus_all_other_coverage_union": versus_all,
                }
                exclusive_probabilities[candidate] = exclusive
            family_summary, family_exclusive = _exclusive_summary(
                detected_by_arm,
                fault_ids,
                candidate_arms=REASONING_CANDIDATES,
                comparator_arms=non_reasoning_arms,
            )
            outcome_report["budgets"][str(fraction)] = {
                "budget_cases": budget,
                "arm_detection": arm_detection,
                "reasoning_candidates": candidate_report,
                "reasoning_candidate_union": family_summary,
            }
            for fault_id in sorted(fault_ids):
                context = {
                    dimension: sorted(values_by_id[fault_id])
                    for dimension, values_by_id in context_features.items()
                    if fault_id in values_by_id
                }
                row: dict[str, Any] = {
                    "outcome": outcome,
                    "budget_fraction": fraction,
                    "budget_cases": budget,
                    "fault_id": fault_id,
                    **{
                        f"{arm}_detection_probability": probabilities[fault_id]
                        for arm, probabilities in arm_probabilities.items()
                    },
                    **{
                        f"{candidate}_only_vs_non_reasoning_probability": (
                            exclusive_probabilities[candidate][fault_id]
                        )
                        for candidate in REASONING_CANDIDATES
                    },
                    **{
                        f"{candidate}_only_vs_{comparator}_probability": (
                            probabilities[fault_id]
                        )
                        for (
                            candidate,
                            comparator,
                        ), probabilities in reasoning_pairwise_probabilities.items()
                    },
                    "reasoning_candidate_union_only_probability": family_exclusive[
                        fault_id
                    ],
                    "context": json.dumps(context, sort_keys=True),
                }
                csv_rows.append(row)
        report["outcomes"][outcome] = outcome_report
    return report, csv_rows


def _curve_report(
    analysis: DatasetAnalysis,
    outcomes: Iterable[str],
    *,
    points: int,
    value_budgets: tuple[float, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fractions = _curve_fractions(len(analysis.case_ids), points)
    report: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        units = analysis.outcomes[outcome]
        arm_results: dict[str, Any] = {}
        for arm, arm_orders in analysis.orders.items():
            curve = _curve_summary(arm_orders, units, fractions)
            arm_results[arm] = curve
            for index, fraction in enumerate(curve["budget_fraction"]):
                rows.append(
                    {
                        "outcome": outcome,
                        "arm": arm,
                        "budget_fraction": fraction,
                        "budget_cases": curve["budget_cases"][index],
                        "mean_faults": curve["mean_faults"][index],
                        "p2_5_faults": curve["p2_5_faults"][index],
                        "p97_5_faults": curve["p97_5_faults"][index],
                        "total_faults": len(units),
                    }
                )
        at_budget: dict[str, Any] = {}
        for fraction in value_budgets:
            key = str(fraction)
            probabilities = {
                arm: _probabilities_at_budget(orders, units, fraction)
                for arm, orders in analysis.orders.items()
            }
            random_mean = fmean(probabilities["random"].values()) * len(units)
            at_budget[key] = {
                arm: {
                    "mean_faults": fmean(values.values()) * len(units),
                    "mean_recall": fmean(values.values()),
                    "delta_faults_vs_random": (
                        fmean(values.values()) * len(units) - random_mean
                    ),
                }
                for arm, values in probabilities.items()
            }
        report[outcome] = {
            "n_faults": len(units),
            "apfd": {
                arm: _distribution([_fault_unit_apfd(order, units) for order in orders])
                for arm, orders in analysis.orders.items()
            },
            "curves": arm_results,
            "at_budget": at_budget,
        }
    return report, rows


def _program_features(program: TableReasoningProgram) -> dict[str, set[str]]:
    stage = program.primary
    return {
        "z1_category": {component.category for component in stage.components},
        "z1_predicate": {component.predicate for component in stage.components},
        "z2_role": {clause.role for clause in stage.clauses},
        "z3_pair": {stage.decision.pair},
        "continuation": {"present" if program.continuations else "absent"},
        "reasoning_signature": {
            " & ".join(
                sorted(
                    item.removeprefix("table:path=")
                    for item in primary_reasoning_items(program)
                    if item.startswith("table:path=")
                )
            )
            or f"unobserved>{stage.decision.pair}"
        },
    }


def _configure_matplotlib() -> None:
    config_dir = Path("/tmp") / f"alpasim-reasoning-coverage-mpl-{os.getuid()}"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)


def _categorical_distribution(
    values_by_id: dict[str, set[str]], ids: Iterable[str]
) -> list[dict[str, Any]]:
    selected = list(ids)
    counter = Counter(
        value for case_id in selected for value in values_by_id.get(case_id, set())
    )
    return [
        {
            "value": value,
            "count": count,
            "fraction": count / len(selected) if selected else 0.0,
        }
        for value, count in counter.most_common()
    ]


def _concentration(signatures: Iterable[str]) -> dict[str, Any]:
    counts = Counter(signatures)
    total = sum(counts.values())
    probabilities = np.asarray([count / total for count in counts.values()])
    entropy = float(-(probabilities * np.log(probabilities)).sum()) if total else 0.0
    return {
        "n_faults": total,
        "n_unique_signatures": len(counts),
        "largest_signature_count": max(counts.values(), default=0),
        "largest_signature_fraction": (
            max(counts.values(), default=0) / total if total else 0.0
        ),
        "effective_number_of_signatures": math.exp(entropy) if total else 0.0,
        "normalized_entropy": (
            entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
        ),
        "singleton_signature_fraction": (
            sum(count == 1 for count in counts.values()) / len(counts)
            if counts
            else 0.0
        ),
        "top_signatures": [
            {"signature": signature, "count": count, "fraction": count / total}
            for signature, count in counts.most_common(10)
        ],
    }


def _value_by_stratum(
    analysis: DatasetAnalysis,
    outcome: str,
    features: dict[str, dict[str, set[str]]],
    *,
    budget: float,
    candidate: str = "reasoning_path",
) -> list[dict[str, Any]]:
    units = analysis.outcomes[outcome]
    probabilities = {
        arm: _probabilities_at_budget(orders, units, budget)
        for arm, orders in analysis.orders.items()
    }
    membership: dict[tuple[str, str], set[str]] = defaultdict(set)
    for fault_id in units:
        for dimension, values_by_id in features.items():
            for value in values_by_id.get(fault_id, set()):
                membership[(dimension, value)].add(fault_id)
    rows = []
    for (dimension, value), fault_ids in membership.items():
        candidate_recall = fmean(probabilities[candidate][item] for item in fault_ids)
        baseline_recalls = {
            arm: fmean(arm_values[item] for item in fault_ids)
            for arm, arm_values in probabilities.items()
            if arm != candidate
        }
        rows.append(
            {
                "dimension": dimension,
                "value": value,
                "n_faults": len(fault_ids),
                "reasoning_recall": candidate_recall,
                "delta_vs_random": candidate_recall - baseline_recalls["random"],
                "delta_vs_trajectory_grid": candidate_recall
                - baseline_recalls["trajectory_grid"],
                "delta_vs_text_bigram": candidate_recall
                - baseline_recalls["text_bigram"],
                **(
                    {
                        "delta_vs_specification": candidate_recall
                        - baseline_recalls["specification"]
                    }
                    if "specification" in baseline_recalls
                    else {}
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (-row["delta_vs_trajectory_grid"], -row["n_faults"]),
    )


def _fault_advantage(
    analysis: DatasetAnalysis,
    outcome: str,
    *,
    budget: float,
    candidate: str = "reasoning_path",
) -> dict[str, Any]:
    units = analysis.outcomes[outcome]
    probabilities = {
        arm: _probabilities_at_budget(orders, units, budget)
        for arm, orders in analysis.orders.items()
    }
    comparisons: dict[str, Any] = {}
    for baseline in analysis.orders:
        if baseline == candidate:
            continue
        margins = {
            fault_id: probabilities[candidate][fault_id]
            - probabilities[baseline][fault_id]
            for fault_id in units
        }
        comparisons[baseline] = {
            "n_reasoning_advantage_ge_0_25": sum(
                margin >= 0.25 for margin in margins.values()
            ),
            "n_reasoning_advantage_ge_0_50": sum(
                margin >= 0.5 for margin in margins.values()
            ),
            "n_baseline_advantage_ge_0_25": sum(
                margin <= -0.25 for margin in margins.values()
            ),
            "mean_probability_delta": fmean(margins.values()),
            "top_reasoning_advantaged_faults": [
                {
                    "fault_id": fault_id,
                    "reasoning_probability": probabilities[candidate][fault_id],
                    "baseline_probability": probabilities[baseline][fault_id],
                    "delta": margin,
                }
                for fault_id, margin in sorted(
                    margins.items(), key=lambda item: (-item[1], item[0])
                )[:20]
            ],
        }
    comparator_arms = [
        arm for arm in analysis.orders if arm not in {candidate, "random"}
    ]
    exclusive = []
    for fault_id in units:
        candidate_probability = probabilities[candidate][fault_id]
        other_max = max(probabilities[arm][fault_id] for arm in comparator_arms)
        if candidate_probability >= 0.75 and other_max <= 0.25:
            exclusive.append(
                {
                    "fault_id": fault_id,
                    "reasoning_probability": candidate_probability,
                    "max_other_probability": other_max,
                }
            )
    return {
        "budget_fraction": budget,
        "comparisons": comparisons,
        "strict_reasoning_only_faults": exclusive,
    }


def _unsafe_mode(entry: dict[str, Any]) -> str:
    text = str(entry.get("planned_trajectory_safety_justification") or "").lower()
    rules = (
        (
            "agent or obstacle collision/clearance",
            r"crash|hitting|too close|how close|leading vehicle",
        ),
        (
            "wrong-way or opposing-lane maneuver",
            r"oppos(?:ing|ite)|oncoming|u-turn|wrong .*lane",
        ),
        (
            "road-edge or off-road excursion",
            r"shoulder|sidewalk|curb|tree lawn|guardrail|side of the road",
        ),
        ("traffic-control violation", r"red traffic|stop sign|right-of-way"),
        ("wrong lane or merge facility", r"entry merge lane|wrong lane"),
        ("other collision risk", r"collision"),
    )
    return next(
        (name for name, pattern in rules if re.search(pattern, text)), "unspecified"
    )


def _inconsistency_mode(entry: dict[str, Any]) -> str:
    raw = entry.get("justification") or entry.get(
        "cot_action_consistency_justification"
    )
    values = raw if isinstance(raw, list) else [raw]
    text = " ".join(str(value) for value in values if value).lower()
    rules = (
        (
            "premature or missing continuation",
            r"early in the plan|immediately|wait for a gap",
        ),
        (
            "maneuver magnitude mismatch",
            r"unnoticeable|nudge.*lane change|lane change.*nudge",
        ),
        (
            "longitudinal control mismatch",
            r"instead of stopping|does not clearly decelerate|only slows|"
            r"did not reach full stop|no significant deceleration|accelerat",
        ),
        (
            "lateral maneuver/direction mismatch",
            r"lane[- ]change|changes? (?:left|right)|(?:left|right) turn|"
            r"turn(?:s|ing)? (?:left|right)|u-turn|rightward|leftward|merge|"
            r"leaving .*lane|nudge was",
        ),
    )
    return next(
        (name for name, pattern in rules if re.search(pattern, text)), "other mismatch"
    )


def _nurec_values(entry: dict[str, Any]) -> set[str]:
    scene = entry.get("labels", {}).get("nurec_scene", {})
    result = set()
    for field in SCENE_FIELDS:
        raw = scene.get(field)
        values = raw if isinstance(raw, list) else [raw]
        result.update(f"{field}={value}" for value in values if value is not None)
    return result


def _risk_table(
    values_by_id: dict[str, set[str]], fault_ids: set[str], all_ids: set[str]
) -> list[dict[str, Any]]:
    overall_rate = len(fault_ids) / len(all_ids)
    rows = []
    all_values = sorted(set().union(*(values_by_id[case_id] for case_id in all_ids)))
    for value in all_values:
        exposed = {case_id for case_id in all_ids if value in values_by_id[case_id]}
        failures = exposed & fault_ids
        rate = len(failures) / len(exposed)
        rows.append(
            {
                "value": value,
                "n_cases": len(exposed),
                "n_faults": len(failures),
                "fault_rate": rate,
                "risk_ratio_vs_cohort": rate / overall_rate if overall_rate else 0.0,
            }
        )
    return sorted(
        rows, key=lambda row: (-row["risk_ratio_vs_cohort"], -row["n_faults"])
    )


def _benchmark_analysis(
    benchmark_path: Path, *, repeats: int, seed: int, multi_round: bool
) -> tuple[DatasetAnalysis, dict[str, Any], dict[str, dict[str, set[str]]]]:
    cases, _ = build_cases(benchmark_path)
    generic, _ = build_benchmark_coverage(cases)
    programs = {
        case.clip_id: compile_table_program(case.chain_of_thought) for case in cases
    }
    coverage = {
        "specification": generic["specification"],
        "visual_stats": generic["visual_stats"],
        "image_grid": generic["image_grid"],
        "trajectory": generic["trajectory"],
        "trajectory_grid": generic["trajectory_grid"],
        "reasoning_path": {
            case_id: frozenset(primary_reasoning_items(program))
            for case_id, program in programs.items()
        },
        "atomic_reasoning_path": {
            case_id: frozenset(atomic_reasoning_path_items(program))
            for case_id, program in programs.items()
        },
        "reasoning_continuation": {
            case_id: frozenset(sequence_reasoning_items(program))
            for case_id, program in programs.items()
        },
        "text_bigram": generic["text_bigram"],
    }
    case_ids = tuple(sorted(programs))
    outcomes = {
        outcome: _single_case_faults(
            case.clip_id for case in cases if case.outcome(outcome)
        )
        for outcome in BENCHMARK_OUTCOMES
    }
    analysis = DatasetAnalysis(
        case_ids=case_ids,
        coverage=coverage,
        orders=_orders(
            case_ids, coverage, repeats=repeats, seed=seed, multi_round=multi_round
        ),
        outcomes=outcomes,
    )

    entries_by_id = {
        str(entry["clip_id"]): entry for entry in _entries(_read_json(benchmark_path))
    }
    program_feature_rows = {
        dimension: {
            case_id: values
            for case_id, program in programs.items()
            for current_dimension, values in _program_features(program).items()
            if current_dimension == dimension
        }
        for dimension in (
            "z1_category",
            "z1_predicate",
            "z2_role",
            "z3_pair",
            "continuation",
            "reasoning_signature",
        )
    }
    context_features = {
        **program_feature_rows,
        "nurec": {
            case_id: _nurec_values(entry) for case_id, entry in entries_by_id.items()
        },
        "unsafe_mode": {
            case_id: {_unsafe_mode(entry)} for case_id, entry in entries_by_id.items()
        },
        "inconsistency_mode": {
            case_id: {_inconsistency_mode(entry)}
            for case_id, entry in entries_by_id.items()
        },
        "source_scene": {
            case_id: {case_id.rsplit("_t", maxsplit=1)[0]} for case_id in case_ids
        },
    }
    characterization = {}
    all_ids = set(case_ids)
    for outcome, units in outcomes.items():
        fault_ids = set(units)
        signature_by_id = program_feature_rows["reasoning_signature"]
        relevant_mode = (
            "unsafe_mode" if outcome == "unsafe_trajectory" else "inconsistency_mode"
        )
        characterization[outcome] = {
            "nurec_risk": _risk_table(context_features["nurec"], fault_ids, all_ids),
            "reasoning_dimensions": {
                dimension: _categorical_distribution(values, fault_ids)
                for dimension, values in program_feature_rows.items()
                if dimension != "reasoning_signature"
            },
            "failure_modes": _categorical_distribution(
                context_features[relevant_mode], fault_ids
            ),
            "reasoning_path_concentration": _concentration(
                next(iter(signature_by_id[fault_id])) for fault_id in fault_ids
            ),
        }
    return analysis, characterization, context_features


def _load_exposures(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = _read_json(path)
    if payload.get("kind") != "alpamayo_causal_obligation_exposures":
        raise ValueError(f"Unexpected exposure artifact: {path}")
    return {row["frame_id"]: row["obligations"] for row in payload["frames"]}


def _offroad_position(metadata_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    first_time_s = row["first_center_offroad_time_s"]
    if first_time_s is None:
        return {}
    metadata = _read_json(metadata_path)
    trajectory = metadata["trajectory_xy_rig_frame"]
    target_us = row["time_now_us"] + round(float(first_time_s) * 1e6)
    point = min(trajectory, key=lambda item: abs(int(item["timestamp_us"]) - target_us))
    lateral = float(point["ry"])
    if lateral > 0.25:
        side = "ego-left"
    elif lateral < -0.25:
        side = "ego-right"
    else:
        side = "small ego-relative lateral offset"
    return {
        "first_offroad_forward_m": float(point["rx"]),
        "first_offroad_lateral_m": lateral,
        "first_offroad_side": side,
    }


def _trajectory_grid_cell(item: str) -> tuple[int, int]:
    prefix = "trajectory_grid:cell="
    if not item.startswith(prefix):
        raise ValueError(f"Unexpected trajectory-grid item: {item!r}")
    coordinates = item.removeprefix(prefix).split(",")
    if len(coordinates) != 2:
        raise ValueError(f"Unexpected trajectory-grid item: {item!r}")
    try:
        return int(coordinates[0]), int(coordinates[1])
    except ValueError as exc:
        raise ValueError(f"Unexpected trajectory-grid item: {item!r}") from exc


def _mean_fault_value_by_cell(
    cells_by_fault: dict[str, frozenset[tuple[int, int]]],
    values_by_fault: dict[str, float],
) -> dict[tuple[int, int], dict[str, float | int]]:
    counts: Counter[tuple[int, int]] = Counter()
    totals: defaultdict[tuple[int, int], float] = defaultdict(float)
    if set(cells_by_fault) != set(values_by_fault):
        raise ValueError("Fault cells and values must contain the same fault IDs")
    for fault_id, cells in cells_by_fault.items():
        for cell in cells:
            counts[cell] += 1
            totals[cell] += values_by_fault[fault_id]
    return {
        cell: {
            "n_positive_frames": count,
            "mean_value": totals[cell] / count,
        }
        for cell, count in counts.items()
    }


def _spatial_outcome_rows(
    analysis: DatasetAnalysis,
    outcome: str,
    cells_by_fault: dict[str, frozenset[tuple[int, int]]],
    exact_locations: Counter[tuple[int, int]],
    *,
    budget: float,
) -> list[dict[str, Any]]:
    units = analysis.outcomes[outcome]
    if set(cells_by_fault) != set(units):
        raise ValueError("Spatial cells must be supplied for every fault unit")
    onset_counts = Counter(cell for cells in cells_by_fault.values() for cell in cells)
    probabilities = {
        arm: _probabilities_at_budget(orders, units, budget)
        for arm, orders in analysis.orders.items()
        if arm in {"trajectory_grid", "reasoning_path"}
    }
    cell_detection = {
        arm: _mean_fault_value_by_cell(cells_by_fault, values)
        for arm, values in probabilities.items()
    }
    rows = []
    for x_index, y_index in sorted(set(onset_counts) | set(exact_locations)):
        trajectory_probability = (
            cell_detection["trajectory_grid"]
            .get((x_index, y_index), {})
            .get("mean_value")
        )
        reasoning_probability = (
            cell_detection["reasoning_path"]
            .get((x_index, y_index), {})
            .get("mean_value")
        )
        rows.append(
            {
                "outcome": outcome,
                "x_index": x_index,
                "y_index": y_index,
                "forward_center_m": x_index * DEFAULT_CELL_SIZE_M,
                "lateral_center_m": y_index * DEFAULT_CELL_SIZE_M,
                "n_fault_units": onset_counts[(x_index, y_index)],
                "n_exact_first_fault_locations": exact_locations[(x_index, y_index)],
                "trajectory_grid_detection_probability": trajectory_probability,
                "reasoning_path_detection_probability": reasoning_probability,
                "reasoning_minus_trajectory_grid": (
                    reasoning_probability - trajectory_probability
                    if reasoning_probability is not None
                    and trajectory_probability is not None
                    else None
                ),
            }
        )
    return rows


def _benchmark_spatial_grid_report(
    analysis: DatasetAnalysis, *, budget: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Map the 150-case human fault labels onto their emitted plan cells."""
    cells_by_case = {
        case_id: frozenset(_trajectory_grid_cell(item) for item in items)
        for case_id, items in analysis.coverage["trajectory_grid"].items()
    }
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "coordinate_frame": "ego_rig_frame",
        "cell_size_m": DEFAULT_CELL_SIZE_M,
        "horizon_s": DEFAULT_HORIZON_S,
        "budget_fraction": budget,
        "fault_cell_semantics": (
            "Each human-labeled fault case contributes every 2 m cell crossed by "
            "its emitted 3 s plan. Coverage values are the mean case-detection "
            "probability among fault cases crossing the cell."
        ),
        "outcomes": {},
    }
    for outcome in BENCHMARK_OUTCOMES:
        units = analysis.outcomes[outcome]
        cells_by_fault = {
            fault_id: cells_by_case[next(iter(case_ids))]
            for fault_id, case_ids in units.items()
        }
        outcome_rows = _spatial_outcome_rows(
            analysis, outcome, cells_by_fault, Counter(), budget=budget
        )
        rows.extend(outcome_rows)
        summary["outcomes"][outcome] = {
            "n_fault_cases": len(units),
            "n_cells_crossed_by_fault_cases": len(
                set().union(*cells_by_fault.values())
            ),
            "exact_fault_location": (
                "unavailable: the human labels classify the case but do not retain "
                "a fault waypoint"
            ),
        }
    return summary, rows


def _spatial_grid_report(
    analysis: DatasetAnalysis,
    labels_path: Path,
    cache_path: Path,
    *,
    budget: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Map positive frames and prioritization value onto trajectory-grid cells."""
    label_rows = {row["frame_id"]: row for row in _read_json(labels_path)["frames"]}
    cache = _read_json(cache_path)
    metadata_by_frame = {
        row["frame_id"]: Path(path)
        for row, path in zip(cache["frames"], cache["metadata_paths"], strict=True)
    }
    cells_by_frame = {
        frame_id: frozenset(_trajectory_grid_cell(item) for item in items)
        for frame_id, items in analysis.coverage["trajectory_grid"].items()
    }
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "coordinate_frame": "ego_rig_frame",
        "cell_size_m": DEFAULT_CELL_SIZE_M,
        "horizon_s": DEFAULT_HORIZON_S,
        "budget_fraction": budget,
        "fault_cell_semantics": (
            "Each positive frame is anchored to its own emitted plan. Path counts "
            "include every 2 m cell crossed by that plan; coverage values are the "
            "mean frame-detection probability among plans crossing the cell."
        ),
        "outcomes": {},
    }
    for outcome in PLANNED_OUTCOMES:
        units = analysis.outcomes[outcome]
        cells_by_fault = {frame_id: cells_by_frame[frame_id] for frame_id in units}
        exact_locations: Counter[tuple[int, int]] = Counter()
        if outcome == OFFROAD_OUTCOME:
            for frame_id in units:
                position = _offroad_position(
                    metadata_by_frame[frame_id], label_rows[frame_id]
                )
                exact_locations[
                    trajectory_grid_cell(
                        position["first_offroad_forward_m"],
                        position["first_offroad_lateral_m"],
                    )
                ] += 1

        outcome_rows = _spatial_outcome_rows(
            analysis, outcome, cells_by_fault, exact_locations, budget=budget
        )
        rows.extend(outcome_rows)
        summary["outcomes"][outcome] = {
            "n_positive_frames": len(units),
            "n_cells_crossed_by_positive_frame_plans": len(
                set().union(*cells_by_fault.values())
            ),
            "exact_fault_location": (
                "first emitted waypoint whose ego center is outside the drivable union"
                if outcome == OFFROAD_OUTCOME
                else "unavailable: the stored collision metric does not retain the collision waypoint"
            ),
            "n_cells_with_exact_first_fault_locations": len(exact_locations),
        }
    return summary, rows


def _scene_concentration(
    frame_ids: Iterable[str], rows: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    result = _concentration(rows[frame_id]["scene_id"] for frame_id in frame_ids)
    return {
        "n_scenes_with_fault": result["n_unique_signatures"],
        "largest_scene_fraction": result["largest_signature_fraction"],
        "effective_number_of_scenes": result["effective_number_of_signatures"],
        "normalized_entropy": result["normalized_entropy"],
        "top_scenes": [
            {
                "scene_id": item["signature"],
                "count": item["count"],
                "fraction": item["fraction"],
            }
            for item in result["top_signatures"]
        ],
    }


def _planned_analysis(
    root: Path,
    benchmark_path: Path,
    labels_path: Path,
    cache_path: Path,
    exposures_path: Path,
    *,
    repeats: int,
    seed: int,
    workers: int,
    multi_round: bool,
) -> tuple[
    DatasetAnalysis,
    dict[str, Any],
    dict[str, dict[str, set[str]]],
    dict[str, Any],
]:
    frames, _ = _load_or_extract_frames(root, cache_path, workers)
    development_scenes = _development_scene_ids(benchmark_path)
    label_payload = _read_json(labels_path)
    label_by_id = {row["frame_id"]: row for row in label_payload["frames"]}
    valid_frames = [
        frame
        for frame in frames
        if frame.scene_id not in development_scenes
        and label_by_id[frame.frame_id]["valid"]
    ]
    if not valid_frames:
        raise ValueError("No valid source-disjoint planned-safety frames")
    case_ids = tuple(sorted(frame.frame_id for frame in valid_frames))
    table_programs_by_text = {
        frame.program.text: compile_table_program(frame.program.text)
        for frame in valid_frames
    }
    table_programs = {
        frame.frame_id: table_programs_by_text[frame.program.text]
        for frame in valid_frames
    }
    coverage = {
        "trajectory_grid": {
            frame.frame_id: frozenset(trajectory_grid_items(frame.metadata_path))
            for frame in valid_frames
        },
        "reasoning_path": {
            frame.frame_id: frozenset(
                primary_reasoning_items(table_programs[frame.frame_id])
            )
            for frame in valid_frames
        },
        "atomic_reasoning_path": {
            frame.frame_id: frozenset(
                atomic_reasoning_path_items(table_programs[frame.frame_id])
            )
            for frame in valid_frames
        },
        "reasoning_continuation": {
            frame.frame_id: frozenset(
                sequence_reasoning_items(table_programs[frame.frame_id])
            )
            for frame in valid_frames
        },
        "text_bigram": {
            frame.frame_id: frozenset(lexical_items(frame.program))
            for frame in valid_frames
        },
    }
    rows = []
    frame_by_id = {frame.frame_id: frame for frame in valid_frames}
    for frame_id in case_ids:
        row = dict(label_by_id[frame_id])
        row["scene_id"] = frame_by_id[frame_id].scene_id
        rows.append(row)
    rows_by_id = {row["frame_id"]: row for row in rows}
    failures = {
        COLLISION_OUTCOME: {
            row["frame_id"]
            for row in rows
            if row["open_loop_collision_with_recorded_agents"]
        },
        OFFROAD_OUTCOME: {
            row["frame_id"] for row in rows if row["planned_center_offroad"]
        },
    }
    outcomes = {
        outcome: _single_case_faults(frame_ids)
        for outcome, frame_ids in failures.items()
    }
    analysis = DatasetAnalysis(
        case_ids=case_ids,
        coverage=coverage,
        orders=_orders(
            case_ids, coverage, repeats=repeats, seed=seed, multi_round=multi_round
        ),
        outcomes=outcomes,
    )

    exposures = _load_exposures(exposures_path)
    scene_details = {row["scene_id"]: row for row in label_payload["inputs"]["scenes"]}
    program_features_by_frame = {
        frame_id: _program_features(program)
        for frame_id, program in table_programs.items()
    }
    context_features: dict[str, dict[str, set[str]]] = {
        dimension: {}
        for dimension in (
            "z1_category",
            "z1_predicate",
            "z2_role",
            "z3_pair",
            "continuation",
            "reasoning_signature",
            "reasoning_entity",
            "road_context",
            "actor_group",
            "obligation_family",
            "distance_band",
            "departure_side",
            "local_lane_count_band",
            "source_scene",
        )
    }
    valid_scene_ids = {frame.scene_id for frame in valid_frames}
    lane_counts = [
        int(scene_details[scene_id]["n_drivable_lanes"])
        for scene_id in sorted(valid_scene_ids)
    ]
    lane_edges = tuple(
        float(value) for value in np.quantile(lane_counts, (1.0 / 3.0, 2.0 / 3.0))
    )
    for outcome, units in outcomes.items():
        for frame_id in units:
            for dimension in (
                "z1_category",
                "z1_predicate",
                "z2_role",
                "z3_pair",
                "continuation",
                "reasoning_signature",
            ):
                context_features[dimension][frame_id] = set(
                    program_features_by_frame[frame_id][dimension]
                )
            predicates = program_features_by_frame[frame_id]["z1_predicate"]
            context_features["reasoning_entity"][frame_id] = (
                predicates & ENTITY_PREDICATES
            ) or {"no_explicit_entity_predicate"}
            context_features["road_context"][frame_id] = {
                ROAD_CONTEXT_BY_PREDICATE[predicate]
                for predicate in predicates
                if predicate in ROAD_CONTEXT_BY_PREDICATE
            } or {"no_explicit_road-context predicate"}
            obligations = exposures.get(frame_id, [])
            context_features["actor_group"][frame_id] = {
                str(obligation["actor_group"]) for obligation in obligations
            } or {"no_high-confidence_actor_obligation"}
            context_features["obligation_family"][frame_id] = {
                str(obligation["family"]) for obligation in obligations
            } or {"no_high-confidence_actor_obligation"}
            context_features["distance_band"][frame_id] = {
                str(obligation["distance_band"]) for obligation in obligations
            } or {"no_high-confidence_actor_obligation"}
            scene_id = rows_by_id[frame_id]["scene_id"]
            context_features["source_scene"][frame_id] = {scene_id}
            lane_count = int(scene_details[scene_id]["n_drivable_lanes"])
            lane_quantile = int(np.searchsorted(lane_edges, lane_count, side="right"))
            lane_band = ("low", "medium", "high")[lane_quantile]
            context_features["local_lane_count_band"][frame_id] = {lane_band}
            if outcome == OFFROAD_OUTCOME:
                position = _offroad_position(
                    frame_by_id[frame_id].metadata_path, rows_by_id[frame_id]
                )
                context_features["departure_side"][frame_id] = {
                    str(position["first_offroad_side"])
                }

    characterization: dict[str, Any] = {}
    for outcome, units in outcomes.items():
        signatures = [
            next(iter(program_features_by_frame[frame_id]["reasoning_signature"]))
            for frame_id in units
        ]
        outcome_context = {
            dimension: _categorical_distribution(values, units)
            for dimension, values in context_features.items()
            if any(frame_id in values for frame_id in units)
        }
        characterization[outcome] = {
            "n_positive_frames": len(units),
            "scene_concentration": _scene_concentration(units, rows_by_id),
            "local_map_lane_polygon_count_tertiles": {
                "lower_edge": lane_edges[0],
                "upper_edge": lane_edges[1],
                "unit": "lane polygons in the extracted local map, not lanes at ego",
            },
            "reasoning_path_concentration": _concentration(signatures),
            "context": outcome_context,
        }
        if outcome == OFFROAD_OUTCOME:
            offroad_rows = [rows_by_id[frame_id] for frame_id in units]
            positions = [
                _offroad_position(frame_by_id[row["frame_id"]].metadata_path, row)
                for row in offroad_rows
            ]
            characterization[outcome]["offroad_geometry"] = {
                "first_offroad_time_s": {
                    "median": float(
                        np.median(
                            [row["first_center_offroad_time_s"] for row in offroad_rows]
                        )
                    ),
                    "p10": float(
                        np.quantile(
                            [
                                row["first_center_offroad_time_s"]
                                for row in offroad_rows
                            ],
                            0.1,
                        )
                    ),
                    "p90": float(
                        np.quantile(
                            [
                                row["first_center_offroad_time_s"]
                                for row in offroad_rows
                            ],
                            0.9,
                        )
                    ),
                },
                "first_offroad_forward_m": {
                    "median": float(
                        np.median([row["first_offroad_forward_m"] for row in positions])
                    ),
                    "p10": float(
                        np.quantile(
                            [row["first_offroad_forward_m"] for row in positions], 0.1
                        )
                    ),
                    "p90": float(
                        np.quantile(
                            [row["first_offroad_forward_m"] for row in positions], 0.9
                        )
                    ),
                },
                "max_center_distance_outside_m": {
                    "median": float(
                        np.median(
                            [
                                row["max_center_distance_outside_drivable_area_m"]
                                for row in offroad_rows
                            ]
                        )
                    ),
                    "p90": float(
                        np.quantile(
                            [
                                row["max_center_distance_outside_drivable_area_m"]
                                for row in offroad_rows
                            ],
                            0.9,
                        )
                    ),
                },
            }
    text_bigram_diagnostic = _planned_text_bigram_diagnostic(
        analysis,
        {frame.frame_id: frame.program.text for frame in valid_frames},
        failures,
        repeats=repeats,
        seed=seed,
        multi_round=multi_round,
    )
    return analysis, characterization, context_features, text_bigram_diagnostic


def _advantage_report(
    analysis: DatasetAnalysis,
    context_features: dict[str, dict[str, set[str]]],
    outcomes: Iterable[str],
) -> dict[str, Any]:
    result = {}
    for outcome in outcomes:
        result[outcome] = {}
        for budget in ADVANTAGE_BUDGETS:
            result[outcome][str(budget)] = {
                "fault_level": _fault_advantage(analysis, outcome, budget=budget),
                "strata": _value_by_stratum(
                    analysis, outcome, context_features, budget=budget
                ),
            }
    return result


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(
    curve_report: dict[str, Any],
    coverage_report: dict[str, dict[str, Any]],
    outcomes: tuple[str, ...],
    path: Path,
    *,
    max_budget_fraction: float,
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13.5, 8.0))
    grid = figure.add_gridspec(
        2, len(outcomes), height_ratios=(0.9, 1.2), hspace=0.32, wspace=0.18
    )
    coverage_axis = figure.add_subplot(grid[0, :])
    fault_axes = [figure.add_subplot(grid[1, index]) for index in range(len(outcomes))]
    n_cases = next(iter(coverage_report.values()))["budget_cases"][-1]

    coverage_axis.plot(
        [],
        [],
        label=f"{DISPLAY_NAMES['random']} — no coverage representation",
        color=COLORS["random"],
        linewidth=2.0,
    )
    saturation_markers: list[tuple[float, float, float, str]] = []
    for arm, coverage in coverage_report.items():
        x = np.asarray(coverage["budget_fraction"]) * 100.0
        mask = x <= max_budget_fraction * 100.0 + 1e-9
        mean = np.asarray(coverage["mean_coverage"]) * 100.0
        low = np.asarray(coverage["p2_5_coverage"]) * 100.0
        high = np.asarray(coverage["p97_5_coverage"]) * 100.0
        saturation = coverage["saturation_cases"]
        median_cases = saturation["median"]
        median_percent = median_cases / n_cases * 100.0
        low_percent = saturation["p2_5"] / n_cases * 100.0
        high_percent = saturation["p97_5"] / n_cases * 100.0
        case_label = (
            f"{int(median_cases)}"
            if median_cases.is_integer()
            else f"{median_cases:.1f}"
        )
        coverage_axis.plot(
            x[mask],
            mean[mask],
            label=(
                f"{DISPLAY_NAMES[arm]} — median full at {case_label}/{n_cases} "
                f"({coverage['n_observed_items']} items)"
            ),
            color=COLORS[arm],
            linewidth=2.0,
        )
        coverage_axis.fill_between(
            x[mask], low[mask], high[mask], color=COLORS[arm], alpha=0.08
        )
        saturation_markers.append((median_percent, low_percent, high_percent, arm))

    all_axes = [coverage_axis, *fault_axes]
    for median_percent, low_percent, high_percent, arm in saturation_markers:
        if median_percent > max_budget_fraction * 100.0 + 1e-9:
            continue
        for axis in all_axes:
            if high_percent > low_percent:
                axis.axvspan(
                    low_percent,
                    high_percent,
                    color=COLORS[arm],
                    alpha=0.035,
                    linewidth=0.0,
                )
            axis.axvline(
                median_percent,
                color=COLORS[arm],
                linestyle="--",
                linewidth=1.0,
                alpha=0.55,
            )

    coverage_axis.axhline(100.0, color="black", linewidth=0.7, alpha=0.35)
    coverage_axis.set_title(
        f"Coverage of cohort-observed items (N={n_cases} cases)\n"
        "Dashed lines mark median saturation; pale bands show the 95% tie-break range"
    )
    coverage_axis.set_xlabel("Cases selected (% of cohort)")
    coverage_axis.set_ylabel("Observed coverage (%)")
    coverage_axis.grid(alpha=0.25)
    coverage_axis.set_xlim(0.0, max_budget_fraction * 100.0)
    coverage_axis.set_ylim(0.0, 103.0)

    for axis, outcome in zip(fault_axes, outcomes, strict=True):
        outcome_report = curve_report[outcome]
        for arm, curve in outcome_report["curves"].items():
            x = np.asarray(curve["budget_fraction"]) * 100.0
            mask = x <= max_budget_fraction * 100.0 + 1e-9
            mean = np.asarray(curve["mean_faults"])
            low = np.asarray(curve["p2_5_faults"])
            high = np.asarray(curve["p97_5_faults"])
            axis.plot(
                x[mask],
                mean[mask],
                color=COLORS[arm],
                linewidth=2.0,
            )
            axis.fill_between(
                x[mask], low[mask], high[mask], color=COLORS[arm], alpha=0.08
            )
        axis.set_title(f"{OUTCOME_NAMES[outcome]} (N={outcome_report['n_faults']})")
        axis.set_xlabel("Cases selected (% of cohort)")
        axis.set_ylabel("Faults detected")
        axis.grid(alpha=0.25)
        axis.set_xlim(0.0, max_budget_fraction * 100.0)
        axis.set_ylim(bottom=0.0)
    handles, labels = coverage_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    figure.subplots_adjust(top=0.80, bottom=0.09, left=0.08, right=0.98)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_value(
    benchmark_curves: dict[str, Any],
    planned_curves: dict[str, Any],
    path: Path,
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        (
            benchmark_curves,
            "inconsistent_reliable_cot",
            OUTCOME_NAMES["inconsistent_reliable_cot"],
            "fault cases",
        ),
        (
            benchmark_curves,
            "unsafe_trajectory",
            OUTCOME_NAMES["unsafe_trajectory"],
            "fault cases",
        ),
        (
            planned_curves,
            COLLISION_OUTCOME,
            OUTCOME_NAMES[COLLISION_OUTCOME],
            "positive frames",
        ),
        (
            planned_curves,
            OFFROAD_OUTCOME,
            OUTCOME_NAMES[OFFROAD_OUTCOME],
            "positive frames",
        ),
    )
    budget = 0.1
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
    for axis, (report, outcome, title, unit) in zip(axes.flat, panels, strict=True):
        values = report[outcome]["at_budget"][str(budget)]
        arms = list(values)
        deltas = [values[arm]["delta_faults_vs_random"] for arm in arms]
        axis.barh(
            [DISPLAY_NAMES[arm] for arm in arms],
            deltas,
            color=[COLORS[arm] for arm in arms],
        )
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"{title} at {budget:.0%} budget")
        axis.set_xlabel(f"Additional {unit} detected vs random")
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_characterization(
    benchmark: dict[str, Any], planned: dict[str, Any], path: Path
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        (
            benchmark["unsafe_trajectory"]["failure_modes"],
            "150: unsafe-trajectory modes",
            "Fault cases",
        ),
        (
            benchmark["inconsistent_reliable_cot"]["failure_modes"],
            "150: CoT/action mismatch modes",
            "Fault cases",
        ),
        (
            planned[COLLISION_OUTCOME]["context"]["actor_group"],
            "Large cohort: collision-frame actor context",
            "Positive frames",
        ),
        (
            planned[OFFROAD_OUTCOME]["context"]["departure_side"],
            "Large cohort: ego-relative plan side at first departure",
            "Positive frames",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for axis, (rows, title, unit) in zip(axes.flat, panels, strict=True):
        selected = rows[:8]
        labels = [str(row["value"]).replace("_", " ") for row in selected][::-1]
        counts = [row["count"] for row in selected][::-1]
        axis.barh(labels, counts, color="#4c78a8")
        axis.set_title(title)
        axis.set_xlabel(unit)
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _spatial_grid_array(
    rows: list[dict[str, Any]],
    outcome: str,
    value: str,
    extent: tuple[int, int, int, int],
) -> np.ndarray:
    min_x, max_x, min_y, max_y = extent
    array = np.full((max_y - min_y + 1, max_x - min_x + 1), np.nan)
    for row in rows:
        if row["outcome"] != outcome or row[value] is None:
            continue
        if value.startswith("n_") and row[value] == 0:
            continue
        array[row["y_index"] - min_y, row["x_index"] - min_x] = row[value]
    return array


def _spatial_grid_extent(
    rows: list[dict[str, Any]],
) -> tuple[int, int, int, int]:
    populated = [
        row
        for row in rows
        if row["n_fault_units"] or row["n_exact_first_fault_locations"]
    ]
    if not populated:
        raise ValueError("Spatial grid has no populated cells")
    return (
        min(row["x_index"] for row in populated),
        max(row["x_index"] for row in populated),
        min(row["y_index"] for row in populated),
        max(row["y_index"] for row in populated),
    )


def _draw_spatial_grid(
    axis: Any,
    values: np.ndarray,
    extent: tuple[int, int, int, int],
    *,
    cmap: Any,
    norm: Any,
    title: str,
) -> Any:
    min_x, max_x, min_y, max_y = extent
    x_edges = (np.arange(min_x, max_x + 2, dtype=float) - 0.5) * DEFAULT_CELL_SIZE_M
    y_edges = (np.arange(min_y, max_y + 2, dtype=float) - 0.5) * DEFAULT_CELL_SIZE_M
    mesh = axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(values),
        cmap=cmap,
        norm=norm,
        edgecolors=(1.0, 1.0, 1.0, 0.32),
        linewidth=0.2,
    )
    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.45)
    axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.45)
    axis.scatter(
        [0.0],
        [0.0],
        marker="^",
        s=34,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Forward from ego at decision time (m)")
    axis.set_ylabel("Left of ego at decision time (m)")
    axis.set_aspect("equal")
    return mesh


def _plot_fault_grid_heatmaps(rows: list[dict[str, Any]], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    extent = _spatial_grid_extent(rows)
    panels = (
        (
            COLLISION_OUTCOME,
            "n_fault_units",
            "Collision-positive frames\nCells crossed by the emitted plan",
            "Positive-frame plans crossing cell",
        ),
        (
            OFFROAD_OUTCOME,
            "n_fault_units",
            "Center-off-road-positive frames\nCells crossed by the emitted plan",
            "Positive-frame plans crossing cell",
        ),
        (
            OFFROAD_OUTCOME,
            "n_exact_first_fault_locations",
            "Center-off-road-positive frames\nCell of first center departure",
            "First departures in cell",
        ),
    )
    arrays = [
        _spatial_grid_array(rows, outcome, value, extent)
        for outcome, value, _, _ in panels
    ]
    path_maximum = max(float(np.nanmax(array)) for array in arrays[:2])
    norms = (
        LogNorm(vmin=1.0, vmax=max(2.0, path_maximum)),
        LogNorm(vmin=1.0, vmax=max(2.0, path_maximum)),
        LogNorm(vmin=1.0, vmax=max(2.0, float(np.nanmax(arrays[2])))),
    )
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f1f3f5")
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), constrained_layout=True)
    for axis, array, norm, (_, _, title, colorbar_label) in zip(
        axes, arrays, norms, panels, strict=True
    ):
        mesh = _draw_spatial_grid(
            axis, array, extent, cmap=cmap, norm=norm, title=title
        )
        figure.colorbar(mesh, ax=axis, shrink=0.78, label=colorbar_label)
    figure.suptitle(
        "Positive frames on the 2 m × 2 m ego-centric trajectory grid\n"
        "First 3 s of each emitted plan; triangle marks the decision-time ego origin. "
        "The collision artifact does not store the exact collision waypoint.",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_benchmark_fault_grid_heatmaps(rows: list[dict[str, Any]], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    extent = _spatial_grid_extent(rows)
    panels = (
        (
            "inconsistent_reliable_cot",
            "Inconsistent reliable CoT\nCells crossed by each labeled case's plan",
        ),
        (
            "unsafe_trajectory",
            "Unsafe trajectory\nCells crossed by each labeled case's plan",
        ),
    )
    arrays = [
        _spatial_grid_array(rows, outcome, "n_fault_units", extent)
        for outcome, _ in panels
    ]
    maximum = max(float(np.nanmax(array)) for array in arrays)
    norm = LogNorm(vmin=1.0, vmax=max(2.0, maximum))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f1f3f5")
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), constrained_layout=True)
    for axis, array, (_, title) in zip(axes, arrays, panels, strict=True):
        mesh = _draw_spatial_grid(
            axis, array, extent, cmap=cmap, norm=norm, title=title
        )
        figure.colorbar(mesh, ax=axis, shrink=0.78, label="Fault cases crossing cell")
    figure.suptitle(
        "150-case benchmark faults on the 2 m × 2 m ego-centric trajectory grid\n"
        "Each human-labeled case contributes its emitted plan's cells; triangle "
        "marks the decision-time ego origin. Exact fault waypoints are not labeled.",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_coverage_grid_heatmaps(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    budget: float,
    outcomes: tuple[str, ...],
    outcome_names: dict[str, str],
    cohort_name: str,
    difference_scale_min_support: int,
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if len(outcomes) != 2:
        raise ValueError("Coverage grid heatmap expects exactly two outcomes")
    extent = _spatial_grid_extent(rows)
    value_keys = (
        "trajectory_grid_detection_probability",
        "reasoning_path_detection_probability",
        "reasoning_minus_trajectory_grid",
    )
    probability_cmap = plt.get_cmap("viridis").copy()
    probability_cmap.set_bad("#f1f3f5")
    difference_cmap = plt.get_cmap("RdBu").copy()
    difference_cmap.set_bad("#f1f3f5")
    supported_differences = np.asarray(
        [
            abs(row["reasoning_minus_trajectory_grid"])
            for row in rows
            if row["reasoning_minus_trajectory_grid"] is not None
            and row["n_fault_units"] >= difference_scale_min_support
        ]
    )
    difference_limit = max(
        0.1,
        math.ceil(float(supported_differences.max(initial=0.0)) * 20.0) / 20.0,
    )
    norms = (
        Normalize(vmin=0.0, vmax=1.0),
        Normalize(vmin=0.0, vmax=1.0),
        Normalize(vmin=-difference_limit, vmax=difference_limit),
    )
    cmaps = (probability_cmap, probability_cmap, difference_cmap)
    column_titles = (
        "Trajectory-grid prioritization",
        "Reasoning-path prioritization",
        "Reasoning − trajectory grid",
    )
    figure, axes = plt.subplots(
        2, 3, figsize=(18.0, 9.5), constrained_layout=True, squeeze=False
    )
    meshes: list[list[Any]] = [[], []]
    for row_index, outcome in enumerate(outcomes):
        for column_index, value in enumerate(value_keys):
            array = _spatial_grid_array(rows, outcome, value, extent)
            title = f"{outcome_names[outcome]}\n{column_titles[column_index]}"
            meshes[row_index].append(
                _draw_spatial_grid(
                    axes[row_index, column_index],
                    array,
                    extent,
                    cmap=cmaps[column_index],
                    norm=norms[column_index],
                    title=title,
                )
            )
    figure.colorbar(
        meshes[0][0],
        ax=axes[:, :2],
        shrink=0.72,
        label="Mean probability that the positive frame is detected",
    )
    figure.colorbar(
        meshes[0][2],
        ax=axes[:, 2],
        shrink=0.72,
        extend="both",
        label="Detection-probability difference",
    )
    figure.suptitle(
        f"{cohort_name}: fault detection at a {budget:.0%} review budget, "
        "mapped to 2 m × 2 m cells\n"
        "Each cell averages distinct fault units whose emitted plan crosses that "
        "cell; blue in the difference panels favors reasoning. Difference colors "
        f"clip beyond ±{difference_limit:.0%}.",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_text_bigram_diagnostic(diagnostic: dict[str, Any], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5))
    features = diagnostic["feature_summary"]

    axis = axes[0, 0]
    feature_names = ("cot_words", "unique_bigrams")
    x = np.arange(len(feature_names), dtype=float)
    width = 0.36
    axis.bar(
        x - width / 2,
        [features[name]["non_offroad_mean"] for name in feature_names],
        width,
        label="Other valid frames",
        color="#9e9e9e",
    )
    axis.bar(
        x + width / 2,
        [features[name]["offroad_mean"] for name in feature_names],
        width,
        label="Off-road-positive frames",
        color=COLORS["text_bigram"],
    )
    axis.set_xticks(x, ["CoT words", "Unique bigrams"])
    axis.set_ylabel("Mean per frame")
    axis.set_title("Off-road-positive reasoning text is longer")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)

    axis = axes[0, 1]
    rare = features["support_le_2_bigrams"]
    axis.bar(
        ["Other valid frames", "Off-road-positive frames"],
        [rare["non_offroad_mean"], rare["offroad_mean"]],
        color=["#9e9e9e", COLORS["text_bigram"]],
    )
    axis.set_ylabel("Mean support-≤2 bigrams per frame")
    axis.set_title("Off-road-positive text contains more rare wording")
    axis.grid(axis="y", alpha=0.25)

    axis = axes[1, 0]
    apfd_rows = diagnostic["apfd"][OFFROAD_OUTCOME]
    arms = (
        "random",
        "text_bigram",
        "text_bigram_support2",
        "cot_length",
        "inverse_bigram_support",
    )
    labels = (
        "Random",
        "Raw bigram coverage",
        "Bigrams with support ≥2",
        "CoT length only",
        "Inverse-support score",
    )
    x = np.arange(len(arms), dtype=float)
    axis.bar(x, [apfd_rows[arm] for arm in arms], width, color="#4c78a8")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    axis.set_xticks(x, labels, rotation=24, ha="right")
    axis.set_ylim(0.4, 0.75)
    axis.set_ylabel("Positive-frame APFD")
    axis.set_title("Length and rarity are strong diagnostic signals")
    axis.grid(axis="y", alpha=0.25)

    axis = axes[1, 1]
    phrases = diagnostic["top_offroad_associated_bigrams"][:6][::-1]
    labels = [f"{row['bigram']} (N={row['support']})" for row in phrases]
    values = [row["offroad_fraction"] * 100.0 for row in phrases]
    axis.barh(labels, values, color=COLORS["text_bigram"])
    axis.axvline(
        diagnostic["offroad_prevalence"] * 100.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Cohort prevalence",
    )
    axis.set_xlim(0.0, 100.0)
    axis.set_xlabel("Off-road-positive frames among bigram support (%)")
    axis.set_title("Common bigrams enriched among off-road frames")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.25)

    figure.suptitle(
        "Why text-bigram coverage prioritizes center-off-road outcomes",
        fontsize=16,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_reasoning_strata(
    benchmark_value: dict[str, Any], planned_value: dict[str, Any], path: Path
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        (
            benchmark_value["inconsistent_reliable_cot"]["0.1"]["strata"],
            "150: inconsistent CoT",
            2,
            {"inconsistency_mode", "z1_predicate", "z3_pair"},
        ),
        (
            benchmark_value["unsafe_trajectory"]["0.1"]["strata"],
            "150: unsafe trajectory",
            2,
            {"unsafe_mode", "z1_predicate", "z3_pair"},
        ),
        (
            planned_value[COLLISION_OUTCOME]["0.1"]["strata"],
            "Large cohort: collision-positive frame",
            5,
            {"road_context", "reasoning_entity", "z3_pair", "actor_group"},
        ),
        (
            planned_value[OFFROAD_OUTCOME]["0.1"]["strata"],
            "Large cohort: center-off-road-positive frame",
            5,
            {
                "road_context",
                "reasoning_entity",
                "z1_predicate",
                "z3_pair",
                "actor_group",
                "departure_side",
            },
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    for axis, (rows, title, minimum, dimensions) in zip(axes.flat, panels, strict=True):
        candidates = [
            row
            for row in rows
            if row["n_faults"] >= minimum
            and row["dimension"] in dimensions
            and row["delta_vs_trajectory_grid"] > 0.0
        ]
        selected = []
        seen_values = set()
        for row in candidates:
            if row["value"] in seen_values:
                continue
            selected.append(row)
            seen_values.add(row["value"])
            if len(selected) == 5:
                break
        selected.reverse()
        labels = [
            f"{row['value'].replace('_', ' ')} (N={row['n_faults']})"
            for row in selected
        ]
        y = np.arange(len(selected), dtype=float)
        axis.barh(
            y - 0.18,
            [row["delta_vs_trajectory_grid"] * 100.0 for row in selected],
            height=0.34,
            color=COLORS["trajectory_grid"],
            label="vs trajectory grid",
        )
        axis.barh(
            y + 0.18,
            [row["delta_vs_text_bigram"] * 100.0 for row in selected],
            height=0.34,
            color=COLORS["text_bigram"],
            label="vs text bigram",
        )
        axis.set_yticks(y, labels)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"{title} at 10% budget")
        axis.set_xlabel("Reasoning detection-probability delta (points)")
        axis.grid(axis="x", alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    benchmark, benchmark_characterization, benchmark_features = _benchmark_analysis(
        args.benchmark,
        repeats=args.benchmark_repeats,
        seed=args.seed,
        multi_round=args.multi_round,
    )
    benchmark_curves, benchmark_rows = _curve_report(
        benchmark,
        BENCHMARK_OUTCOMES,
        points=150,
        value_budgets=BENCHMARK_BUDGETS,
    )
    benchmark_coverage = _coverage_report(benchmark, points=150)
    (
        planned,
        planned_characterization,
        planned_features,
        planned_text_bigram_diagnostic,
    ) = _planned_analysis(
        args.frame_root,
        args.benchmark,
        args.planned_labels,
        args.feature_cache,
        args.exposures,
        repeats=args.planned_repeats,
        seed=args.seed,
        workers=args.workers,
        multi_round=args.multi_round,
    )
    planned_curves, planned_rows = _curve_report(
        planned,
        PLANNED_OUTCOMES,
        points=200,
        value_budgets=PLANNED_BUDGETS,
    )
    planned_coverage = _coverage_report(planned, points=2000)
    benchmark_value = _advantage_report(
        benchmark, benchmark_features, BENCHMARK_OUTCOMES
    )
    planned_value = _advantage_report(planned, planned_features, PLANNED_OUTCOMES)
    benchmark_early_discovery, benchmark_early_rows = _early_fault_discovery_report(
        benchmark,
        BENCHMARK_OUTCOMES,
        BENCHMARK_DISCOVERY_BUDGETS,
        (
            "specification",
            "visual_stats",
            "image_grid",
            "trajectory",
            "trajectory_grid",
            "text_bigram",
        ),
        benchmark_features,
        fault_unit="one human-labeled benchmark case",
    )
    planned_early_discovery, planned_early_rows = _early_fault_discovery_report(
        planned,
        PLANNED_OUTCOMES,
        PLANNED_DISCOVERY_BUDGETS,
        (
            "trajectory_grid",
            "text_bigram",
        ),
        planned_features,
        fault_unit="one positive 100 ms planned-safety frame",
    )
    benchmark_spatial_grid, benchmark_spatial_rows = _benchmark_spatial_grid_report(
        benchmark, budget=SPATIAL_GRID_BUDGET
    )
    planned_spatial_grid, planned_spatial_rows = _spatial_grid_report(
        planned,
        args.planned_labels,
        args.feature_cache,
        budget=SPATIAL_GRID_BUDGET,
    )
    report = {
        "schema_version": 1,
        "kind": "reasoning_coverage_experiment_analysis",
        "config": {
            "seed": args.seed,
            "benchmark_repeats": args.benchmark_repeats,
            "planned_repeats": args.planned_repeats,
            "greedy": "multi_round" if args.multi_round else "single_round",
            "reasoning_candidate": "reasoning_path",
            "reasoning_candidates": list(REASONING_CANDIDATES),
            "atomic_reasoning_path_definition": (
                "Only realized primary-stage Z1 component -> Z2 role -> Z3 axis "
                "decision items; auxiliary nodes, details, and edges are excluded"
            ),
            "reasoning_path_definition": (
                "Cumulative primary-stage Z1/Z2/Z3 nodes, component details, "
                "edges, and realized Z1 component -> Z2 role -> Z3 axis decision "
                "items"
            ),
            "reasoning_only_definition": (
                "candidate detection probability >= 0.75 and non-reasoning "
                "comparator-union detection probability <= 0.25 at the stated "
                "budget"
            ),
        },
        "benchmark150": {
            "n_cases": len(benchmark.case_ids),
            "coverage": benchmark_coverage,
            "curves": benchmark_curves,
            "characterization": benchmark_characterization,
            "reasoning_value": benchmark_value,
            "early_fault_discovery": benchmark_early_discovery,
            "spatial_grid": benchmark_spatial_grid,
        },
        "planned_safety": {
            "n_cases": len(planned.case_ids),
            "coverage": planned_coverage,
            "curves": planned_curves,
            "characterization": planned_characterization,
            "reasoning_value": planned_value,
            "early_fault_discovery": planned_early_discovery,
            "text_bigram_diagnostic": planned_text_bigram_diagnostic,
            "spatial_grid": planned_spatial_grid,
        },
        "limitations": [
            "All analyses are retrospective prioritization analyses.",
            "Intervals quantify randomized tie-breaking, not corpus uncertainty.",
            "The 150-case unsafe modes and mismatch modes use transparent keyword coding "
            "of existing human justifications.",
            "Large-cohort actor groups are high-confidence independent obligation contexts, "
            "not confirmed collision counterparts.",
            "Off-road side is measured in the ego rig frame at plan time, not relative to a global map boundary.",
            "The large-cohort fault unit is one positive frame, so a failure lasting many 100 ms decisions "
            "contributes once per frame rather than once per event.",
            "Positive frames from the same scene are statistically dependent; "
            "intervals do not account for that dependence.",
            "Large-cohort image-grid, visual-summary, and trajectory-summary controls are excluded "
            "because exact multi-round ordering is computationally pathological "
            "for their many distinct signatures and small, frequently resetting item universes; "
            "the established trajectory-grid and text-bigram controls remain.",
            "NuRec semantic scene labels are unavailable for the large cohort, "
            "so NuRec specification coverage is evaluated only on the 150-case benchmark.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    _write_csv(benchmark_rows, args.output.with_suffix(".benchmark_curves.csv"))
    _write_csv(planned_rows, args.output.with_suffix(".planned_curves.csv"))
    _write_csv(
        benchmark_early_rows,
        args.output.with_suffix(".benchmark_early_faults.csv"),
    )
    _write_csv(
        planned_early_rows,
        args.output.with_suffix(".planned_early_faults.csv"),
    )
    _write_csv(
        planned_spatial_rows,
        args.output.with_suffix(".spatial_grid.csv"),
    )
    _write_csv(
        benchmark_spatial_rows,
        args.output.with_suffix(".benchmark_spatial_grid.csv"),
    )
    figure_dir = args.figure_dir
    _plot_curves(
        benchmark_curves,
        benchmark_coverage,
        BENCHMARK_OUTCOMES,
        figure_dir / "benchmark150_fault_detection_curves.png",
        max_budget_fraction=1.0,
    )
    _plot_curves(
        planned_curves,
        planned_coverage,
        PLANNED_OUTCOMES,
        figure_dir / "planned_safety_fault_detection_curves.png",
        max_budget_fraction=0.25,
    )
    _plot_value(
        benchmark_curves,
        planned_curves,
        figure_dir / "fault_detection_value_at_10pct.png",
    )
    _plot_characterization(
        benchmark_characterization,
        planned_characterization,
        figure_dir / "fault_characterization.png",
    )
    _plot_reasoning_strata(
        benchmark_value,
        planned_value,
        figure_dir / "reasoning_stratum_value_at_10pct.png",
    )
    _plot_text_bigram_diagnostic(
        planned_text_bigram_diagnostic,
        figure_dir / "text_bigram_offroad_diagnostic.png",
    )
    _plot_fault_grid_heatmaps(
        planned_spatial_rows,
        figure_dir / "planned_safety_fault_grid_heatmaps.png",
    )
    _plot_benchmark_fault_grid_heatmaps(
        benchmark_spatial_rows,
        figure_dir / "benchmark150_fault_grid_heatmaps.png",
    )
    _plot_coverage_grid_heatmaps(
        planned_spatial_rows,
        figure_dir / "planned_safety_coverage_grid_heatmaps.png",
        budget=SPATIAL_GRID_BUDGET,
        outcomes=PLANNED_OUTCOMES,
        outcome_names={
            COLLISION_OUTCOME: "Collision-positive frames",
            OFFROAD_OUTCOME: "Center-off-road-positive frames",
        },
        cohort_name="Source-disjoint planned-safety cohort",
        difference_scale_min_support=5,
    )
    _plot_coverage_grid_heatmaps(
        benchmark_spatial_rows,
        figure_dir / "benchmark150_coverage_grid_heatmaps.png",
        budget=SPATIAL_GRID_BUDGET,
        outcomes=BENCHMARK_OUTCOMES,
        outcome_names=OUTCOME_NAMES,
        cohort_name="150-case benchmark",
        difference_scale_min_support=5,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark.json"))
    parser.add_argument(
        "--frame-root",
        type=Path,
        default=Path(
            "/project/AV/drive-the-thoughts/alpasim/tutorial_alpamayo_ood_hazard_2604"
        ),
    )
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
    parser.add_argument(
        "--exposures",
        type=Path,
        default=Path("runs/reasoning_coverage/ood_hazard_2604.cro_v1_exposures.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/reasoning_coverage/experiment_analysis.json"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("runs/reasoning_coverage/figures"),
    )
    parser.add_argument(
        "--multi-round",
        action="store_true",
        help=(
            "Restart the covered set whenever coverage saturates, so ordering "
            "stays coverage-directed past the point where an arm's observed "
            "item universe is exhausted."
        ),
    )
    parser.add_argument("--benchmark-repeats", type=int, default=200)
    parser.add_argument("--planned-repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.benchmark_repeats <= 0 or args.planned_repeats <= 0:
        raise ValueError("Repeat counts must be positive")
    report = run_analysis(args)
    print(f"Report: {args.output}")
    print(f"Figures: {args.figure_dir}")
    for cohort in ("benchmark150", "planned_safety"):
        print(f"{cohort}: {report[cohort]['n_cases']} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
