# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import pytest
from reasoning_coverage.statistics import (
    paired_apfd_deltas,
    permutation_test_apfd_delta,
    source_group_bootstrap_apfd_delta,
)


def test_paired_apfd_deltas_compare_aligned_repeats() -> None:
    candidate = [
        ["failure-a", "failure-b", "pass-a", "pass-b"],
        ["failure-b", "failure-a", "pass-b", "pass-a"],
    ]
    baseline = [
        ["pass-a", "pass-b", "failure-a", "failure-b"],
        ["pass-b", "pass-a", "failure-b", "failure-a"],
    ]

    result = paired_apfd_deltas(candidate, baseline, {"failure-a", "failure-b"})

    assert result["candidate_apfd"]["mean"] == pytest.approx(0.75)
    assert result["baseline_apfd"]["mean"] == pytest.approx(0.25)
    assert result["delta"]["mean"] == pytest.approx(0.5)
    assert result["paired_deltas"] == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize(
    ("candidate", "baseline", "message"),
    [
        ([[]], [[]], "at least one case"),
        ([["a", "a"]], [["a", "a"]], "duplicate"),
        ([["a", "b"]], [["a", "c"]], "different case universe"),
        ([["a"]], [["a"], ["a"]], "repeat counts"),
    ],
)
def test_paired_apfd_deltas_reject_invalid_orders(
    candidate: list[list[str]], baseline: list[list[str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_apfd_deltas(candidate, baseline, {"a"})


def test_permutation_test_finds_strong_fixed_order_association() -> None:
    case_ids = [f"case-{index}" for index in range(20)]
    failures = set(case_ids[:5])
    candidate = [case_ids]
    baseline = [list(reversed(case_ids))]

    result = permutation_test_apfd_delta(
        candidate,
        baseline,
        failures,
        permutations=999,
        seed=17,
    )

    assert result["observed_mean_apfd_delta"] == pytest.approx(0.75)
    assert result["p_value_greater"] <= 0.01
    assert result["p_value_two_sided"] <= 0.02
    assert result["null_delta"]["mean"] == pytest.approx(0.0, abs=0.03)


def test_permutation_test_rejects_constant_outcomes() -> None:
    order = [["a", "b"]]

    with pytest.raises(ValueError, match="failing and passing"):
        permutation_test_apfd_delta(order, order, {"a", "b"})


def test_source_group_bootstrap_resamples_whole_groups() -> None:
    failures = [f"failure-{index}" for index in range(4)]
    passes = [f"pass-{index}" for index in range(4)]
    candidate = [failures + passes]
    baseline = [passes + failures]
    groups = {
        case_id: f"source-{index}"
        for index, pair in enumerate(zip(failures, passes, strict=True))
        for case_id in pair
    }

    result = source_group_bootstrap_apfd_delta(
        candidate,
        baseline,
        set(failures),
        groups,
        resamples=200,
        seed=9,
    )

    assert result["observed_mean_apfd_delta"] == pytest.approx(0.5)
    assert result["confidence_interval"]["lower"] == pytest.approx(0.5)
    assert result["confidence_interval"]["upper"] == pytest.approx(0.5)
    assert result["n_source_groups"] == 4
    assert result["valid_resamples"] == 200
    assert result["dropped_no_failure_resamples"] == 0
    assert result["limitations"]


def test_source_group_bootstrap_requires_complete_group_mapping() -> None:
    with pytest.raises(ValueError, match="source groups missing"):
        source_group_bootstrap_apfd_delta(
            [["failure", "pass"]],
            [["pass", "failure"]],
            {"failure"},
            {"failure": "source-a"},
            resamples=10,
        )
