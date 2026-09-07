# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "plot_trajectory_grid_path_histogram.py"
)
SPEC = importlib.util.spec_from_file_location("coverage_path_curve", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_reasoning_signature_label = MODULE._reasoning_signature_label
build_curve = MODULE.build_curve


def test_curve_groups_equal_reasoning_coverage_signatures() -> None:
    signature_a = ("table:path=lead>causes>longitudinal:decelerate",)
    signature_b = ("table:z3_pair=maintain_speed|go_straight",)
    records = [
        {"cells": signature_a, "faulty": True},
        {"cells": signature_a, "faulty": False},
        {"cells": signature_b, "faulty": True},
    ]

    curve, summary = build_curve(records)

    assert curve["counts"] == [2, 1]
    assert curve["faults"] == [1, 1]
    assert curve["cumulative_examples"] == [2 / 3, 1.0]
    assert curve["cumulative_faults"] == [0.5, 1.0]
    assert summary["n_paths"] == 2


def test_reasoning_signature_label_prefers_typed_paths() -> None:
    assert (
        _reasoning_signature_label(
            (
                "table:path=lead>causes>longitudinal:decelerate",
                "table:z3_pair=decelerate|go_straight",
            )
        )
        == "lead>causes>longitudinal:decelerate"
    )
