# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import pytest
from reasoning_coverage.planned_safety_evaluation import _outcome_result


def _orders(candidate: list[str]) -> dict[str, list[list[str]]]:
    """One repeat per arm; every comparison baseline must be present."""
    baseline = ["safe-a", "early", "safe-b", "late"]
    return {
        "random": [baseline],
        "trajectory": [baseline],
        "trajectory_grid": [baseline],
        "text_bigram": [baseline],
        "candidate": [candidate],
    }


def test_outcome_result_scores_every_positive_frame_separately() -> None:
    # Both arms select "early" first, so an event-merged score would rate them
    # equally. Counting each positive frame rewards the arm that also pulls
    # "late" forward.
    front = _outcome_result(
        _orders(["early", "late", "safe-a", "safe-b"]), {"early", "late"}
    )
    split = _outcome_result(
        _orders(["early", "safe-a", "safe-b", "late"]), {"early", "late"}
    )

    assert front["n_positive_frames"] == 2
    assert (
        front["arms"]["candidate"]["frame_apfd"]["mean"]
        > split["arms"]["candidate"]["frame_apfd"]["mean"]
    )


def test_outcome_result_reports_paired_deltas_against_each_baseline() -> None:
    result = _outcome_result(_orders(["late", "early", "safe-a", "safe-b"]), {"early"})

    delta = result["comparisons"]["random"]["candidate"]["frame_apfd_delta"]
    assert delta["mean"] == pytest.approx(0.0)
    assert set(result["comparisons"]) == {
        "random",
        "trajectory",
        "trajectory_grid",
        "text_bigram",
    }
