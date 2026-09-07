# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json
import random
from pathlib import Path

from reasoning_coverage.analysis import greedy_coverage_order
from reasoning_coverage.ood_evaluation import (
    _scene_outcomes,
    evaluate_orders,
    lazy_greedy_coverage_order,
)


def test_lazy_greedy_matches_reference_implementation() -> None:
    randomizer = random.Random(9)
    case_ids = [f"case-{index}" for index in range(30)]
    universe = [f"item-{index}" for index in range(20)]
    item_sets = {
        case_id: frozenset(item for item in universe if randomizer.random() < 0.2)
        for case_id in case_ids
    }

    for seed in range(10):
        assert lazy_greedy_coverage_order(case_ids, item_sets, seed) == (
            greedy_coverage_order(case_ids, item_sets, seed)
        )


def _reference_multi_round_order(
    case_ids: list[str], item_sets: dict[str, frozenset[str]], seed: int
) -> list[str]:
    """Readable restatement of multi-round greedy, for cross-checking."""
    remaining = list(case_ids)
    random.Random(seed).shuffle(remaining)
    tie_order = {case_id: index for index, case_id in enumerate(remaining)}
    covered: set[str] = set()
    selected: list[str] = []
    while remaining:
        best = max(
            remaining,
            key=lambda candidate: (
                len(item_sets[candidate] - covered),
                -tie_order[candidate],
            ),
        )
        if not item_sets[best] - covered:
            if not covered:
                break
            covered = set()
            continue
        remaining.remove(best)
        selected.append(best)
        covered.update(item_sets[best])
    selected.extend(case_id for case_id in remaining)
    return selected


def test_multi_round_greedy_matches_reference_implementation() -> None:
    randomizer = random.Random(11)
    case_ids = [f"case-{index}" for index in range(40)]
    universe = [f"item-{index}" for index in range(12)]
    item_sets = {
        case_id: frozenset(item for item in universe if randomizer.random() < 0.25)
        for case_id in case_ids
    }

    for seed in range(10):
        assert lazy_greedy_coverage_order(
            case_ids, item_sets, seed, multi_round=True
        ) == _reference_multi_round_order(case_ids, item_sets, seed)


def test_multi_round_greedy_keeps_covering_after_saturation() -> None:
    item_sets = {
        "wide-a": frozenset({"x", "y"}),
        "wide-b": frozenset({"x", "y"}),
        "narrow-a": frozenset({"x"}),
        "narrow-b": frozenset({"x"}),
    }
    case_ids = list(item_sets)

    for seed in range(10):
        order = lazy_greedy_coverage_order(case_ids, item_sets, seed, multi_round=True)
        assert set(order) == set(case_ids)
        # One "wide" case saturates the universe, so single-round ordering has no
        # basis for the remaining three. Each further round must re-cover "y",
        # which only the other "wide" case supplies.
        assert order[0].startswith("wide")
        assert order[1].startswith("wide")


def test_multi_round_greedy_terminates_with_itemless_cases() -> None:
    item_sets = {
        "covered": frozenset({"x"}),
        "empty-a": frozenset(),
        "empty-b": frozenset(),
    }

    order = lazy_greedy_coverage_order(
        list(item_sets), item_sets, seed=3, multi_round=True
    )

    assert order[0] == "covered"
    assert set(order) == set(item_sets)


def test_scene_outcomes_keep_manual_and_closed_loop_labels_separate(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "aggregate" / "results-summary.json"
    summary_path.parent.mkdir()
    summary_path.write_text(
        json.dumps(
            {
                "rollouts": [{"clipgt_id": "clipgt-test", "passed": True}],
                "metrics_results": [
                    {
                        "collision_at_fault": 0,
                        "offroad": 0,
                        "wrong_lane": 1,
                        "safety_monitor_triggered": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    outcomes = _scene_outcomes(
        tmp_path, {"clipgt-test"}, expected_scene_id="clipgt-test"
    )

    assert outcomes == {
        "closed_loop_failure": False,
        "manual_unsafe_scene": True,
        "closed_loop_safety_event": True,
    }


def test_positive_unlabeled_outcomes_skip_inferential_statistics() -> None:
    orders = {
        "random": [["a", "b", "c"], ["c", "b", "a"]],
        "trajectory": [["b", "a", "c"], ["b", "c", "a"]],
    }

    results = evaluate_orders(
        orders,
        {"known_positive": {"a"}},
        {"known_positive": "positive_unlabeled"},
        {"a": "one", "b": "two", "c": "three"},
        permutation_count=10,
        bootstrap_count=10,
        seed=4,
    )

    comparison = results["known_positive"]["comparisons_to_random"]["trajectory"]
    assert results["known_positive"]["label_scope"] == "positive_unlabeled"
    assert set(comparison) == {"paired_apfd_delta"}
