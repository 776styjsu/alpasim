# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Statistical comparisons for fixed reasoning-coverage orderings.

The functions in this module operate on case orderings that were produced
without consulting the outcome labels.  They deliberately do not construct or
refit a coverage abstraction.  This separation makes the estimand explicit and
prevents an outcome permutation from accidentally changing the prioritization
procedure under test.

An "aligned" pair of repeat orders means that candidate repeat ``i`` and
baseline repeat ``i`` used the same repeat seed or other shared source of
randomness.  Pairing removes repeat-level tie-breaking noise from the APFD
difference.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Hashable, Mapping, Sequence
from statistics import fmean, pstdev
from typing import Any

import numpy as np

from .analysis import apfd


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(fmean(values)),
        "std": float(pstdev(values)),
        "p2_5": float(np.quantile(array, 0.025)),
        "p97_5": float(np.quantile(array, 0.975)),
    }


def _validate_aligned_orders(
    candidate_orders: Sequence[Sequence[str]],
    baseline_orders: Sequence[Sequence[str]],
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...], tuple[str, ...]]:
    if not candidate_orders or not baseline_orders:
        raise ValueError("candidate_orders and baseline_orders must be non-empty")
    if len(candidate_orders) != len(baseline_orders):
        raise ValueError("candidate and baseline repeat counts must match")

    candidate = tuple(tuple(order) for order in candidate_orders)
    baseline = tuple(tuple(order) for order in baseline_orders)
    case_ids = candidate[0]
    if not case_ids:
        raise ValueError("orders must contain at least one case")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("orders may not contain duplicate case IDs")
    universe = set(case_ids)

    for repeat, (candidate_order, baseline_order) in enumerate(
        zip(candidate, baseline, strict=True)
    ):
        for arm, order in (
            ("candidate", candidate_order),
            ("baseline", baseline_order),
        ):
            if len(order) != len(universe) or len(set(order)) != len(order):
                raise ValueError(
                    f"{arm} order at repeat {repeat} contains duplicate or missing cases"
                )
            if set(order) != universe:
                raise ValueError(
                    f"{arm} order at repeat {repeat} has a different case universe"
                )
    return candidate, baseline, case_ids


def _validate_failures(
    failing_ids: Collection[str], case_ids: Sequence[str]
) -> set[str]:
    failures = set(failing_ids)
    if not failures:
        raise ValueError("at least one failing case is required to compute APFD")
    missing = failures - set(case_ids)
    if missing:
        raise ValueError(f"failing IDs absent from orderings: {sorted(missing)}")
    return failures


def _mean_rank_advantage_by_case(
    candidate_orders: Sequence[Sequence[str]],
    baseline_orders: Sequence[Sequence[str]],
    case_ids: Sequence[str],
) -> dict[str, float]:
    """Return mean normalized baseline-rank minus candidate-rank by case."""
    advantages = {case_id: [] for case_id in case_ids}
    size = len(case_ids)
    for candidate_order, baseline_order in zip(
        candidate_orders, baseline_orders, strict=True
    ):
        candidate_ranks = {
            case_id: rank for rank, case_id in enumerate(candidate_order, start=1)
        }
        baseline_ranks = {
            case_id: rank for rank, case_id in enumerate(baseline_order, start=1)
        }
        for case_id in case_ids:
            advantages[case_id].append(
                (baseline_ranks[case_id] - candidate_ranks[case_id]) / size
            )
    return {
        case_id: float(fmean(repeat_values))
        for case_id, repeat_values in advantages.items()
    }


def paired_apfd_deltas(
    candidate_orders: Sequence[Sequence[str]],
    baseline_orders: Sequence[Sequence[str]],
    failing_ids: Collection[str],
) -> dict[str, Any]:
    """Compare APFD for candidate and baseline over aligned repeat orders.

    The percentile range describes variation from randomized tie breaking (or
    whatever generated the supplied repeats).  It is not a confidence interval
    over independently sampled cases or source scenes.
    """
    candidate, baseline, case_ids = _validate_aligned_orders(
        candidate_orders, baseline_orders
    )
    failures = _validate_failures(failing_ids, case_ids)

    candidate_values = [apfd(list(order), failures) for order in candidate]
    baseline_values = [apfd(list(order), failures) for order in baseline]
    if any(value is None for value in (*candidate_values, *baseline_values)):
        raise AssertionError("validated non-empty failures produced undefined APFD")
    candidate_apfd = [float(value) for value in candidate_values if value is not None]
    baseline_apfd = [float(value) for value in baseline_values if value is not None]
    deltas = [
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(
            candidate_apfd, baseline_apfd, strict=True
        )
    ]

    return {
        "n_cases": len(case_ids),
        "n_failures": len(failures),
        "n_repeats": len(candidate),
        "candidate_apfd": _distribution(candidate_apfd),
        "baseline_apfd": _distribution(baseline_apfd),
        "delta": _distribution(deltas),
        "paired_deltas": deltas,
        "interval_scope": "supplied repeat/tie-breaking variation only",
    }


def permutation_test_apfd_delta(
    candidate_orders: Sequence[Sequence[str]],
    baseline_orders: Sequence[Sequence[str]],
    failing_ids: Collection[str],
    *,
    permutations: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Test fixed-order APFD advantage by permuting binary outcome labels.

    Every permutation preserves the observed number of failures.  The supplied
    orders remain fixed, so this is an association test between order rank and
    outcome, not a test that refits coverage or reruns a learned prioritizer.
    The one-sided p-value tests whether the candidate has higher APFD; the
    two-sided value is also reported. In an adaptive exploratory analysis, both
    values are descriptive unless the arm, outcome, and direction were locked
    before outcome inspection.

    Exchangeability of outcome labels across cases is required.  Use a grouped
    or otherwise restricted randomization design when source-scene dependence,
    strata, or sampling weights make unrestricted permutations invalid.
    """
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    candidate, baseline, case_ids = _validate_aligned_orders(
        candidate_orders, baseline_orders
    )
    failures = _validate_failures(failing_ids, case_ids)
    if len(failures) == len(case_ids):
        raise ValueError("permutation test requires both failing and passing cases")

    rank_advantage = _mean_rank_advantage_by_case(candidate, baseline, case_ids)
    observed = float(fmean(rank_advantage[case_id] for case_id in failures))
    randomizer = random.Random(seed)
    null = [
        float(
            fmean(
                rank_advantage[case_id]
                for case_id in randomizer.sample(case_ids, len(failures))
            )
        )
        for _ in range(permutations)
    ]
    greater_count = sum(value >= observed for value in null)
    less_count = sum(value <= observed for value in null)
    two_sided_count = sum(abs(value) >= abs(observed) for value in null)
    denominator = permutations + 1

    return {
        "observed_mean_apfd_delta": observed,
        "n_cases": len(case_ids),
        "n_failures": len(failures),
        "n_repeats": len(candidate),
        "permutations": permutations,
        "seed": seed,
        "p_value_greater": (greater_count + 1) / denominator,
        "p_value_less": (less_count + 1) / denominator,
        "p_value_two_sided": (two_sided_count + 1) / denominator,
        "null_delta": _distribution(null),
        "limitations": [
            "Orders and coverage abstractions are held fixed rather than refit.",
            "Unrestricted permutation assumes outcome-label exchangeability across cases.",
            "The null preserves failure prevalence but not source groups or other strata.",
            "Reported p-values have Monte Carlo error from finite permutations.",
        ],
    }


def source_group_bootstrap_apfd_delta(
    candidate_orders: Sequence[Sequence[str]],
    baseline_orders: Sequence[Sequence[str]],
    failing_ids: Collection[str],
    source_group_by_case: Mapping[str, Hashable],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap the paired APFD delta by resampling source groups.

    The statistic is the case-weighted mean normalized rank advantage among
    failing cases.  This is algebraically equal to candidate APFD minus baseline
    APFD for a common case universe.  Source groups are sampled uniformly with
    replacement, and every failure in a sampled group inherits its group's
    multiplicity.

    This conditional rank bootstrap is appropriate for uncertainty checks on
    fixed, blinded orderings.  It does not rebuild quantile bins, coverage items,
    or greedy orders inside each resample; a full pipeline bootstrap would be
    needed to include those sources of estimation variability.
    """
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    candidate, baseline, case_ids = _validate_aligned_orders(
        candidate_orders, baseline_orders
    )
    failures = _validate_failures(failing_ids, case_ids)

    missing_groups = set(case_ids) - source_group_by_case.keys()
    if missing_groups:
        raise ValueError(
            f"source groups missing for case IDs: {sorted(missing_groups)}"
        )
    groups = list(dict.fromkeys(source_group_by_case[case_id] for case_id in case_ids))
    if len(groups) < 2:
        raise ValueError("at least two source groups are required for bootstrap")

    rank_advantage = _mean_rank_advantage_by_case(candidate, baseline, case_ids)
    failure_values_by_group: dict[Hashable, list[float]] = {
        group: [] for group in groups
    }
    for case_id in failures:
        failure_values_by_group[source_group_by_case[case_id]].append(
            rank_advantage[case_id]
        )

    observed = float(fmean(rank_advantage[case_id] for case_id in failures))
    randomizer = random.Random(seed)
    estimates: list[float] = []
    dropped = 0
    for _ in range(resamples):
        sampled_groups = randomizer.choices(groups, k=len(groups))
        sampled_values = [
            value
            for group in sampled_groups
            for value in failure_values_by_group[group]
        ]
        if not sampled_values:
            dropped += 1
            continue
        estimates.append(float(fmean(sampled_values)))
    if not estimates:
        raise ValueError("no bootstrap resample contained a failing case")

    tail = (1.0 - confidence) / 2.0
    estimate_array = np.asarray(estimates, dtype=float)
    return {
        "observed_mean_apfd_delta": observed,
        "confidence": confidence,
        "confidence_interval": {
            "lower": float(np.quantile(estimate_array, tail)),
            "upper": float(np.quantile(estimate_array, 1.0 - tail)),
            "method": "percentile source-group bootstrap",
        },
        "bootstrap_delta": _distribution(estimates),
        "n_cases": len(case_ids),
        "n_failures": len(failures),
        "n_source_groups": len(groups),
        "n_failure_source_groups": sum(
            bool(failure_values_by_group[group]) for group in groups
        ),
        "requested_resamples": resamples,
        "valid_resamples": len(estimates),
        "dropped_no_failure_resamples": dropped,
        "seed": seed,
        "estimand": "case-weighted mean APFD rank advantage among failures",
        "limitations": [
            "The supplied coverage orderings are fixed and are not refit per resample.",
            "Percentile intervals can be unstable with few source or failure groups.",
            "Source-group IDs must capture all dependence, including adjacent frames.",
            "Resamples with no failing case have undefined APFD and are omitted.",
        ],
    }
