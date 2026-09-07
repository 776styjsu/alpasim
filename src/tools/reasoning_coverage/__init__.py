# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Coverage abstractions and prioritization for Alpamayo reasoning traces."""

from .analysis import (
    BenchmarkCase,
    ReasoningState,
    apfd,
    build_cases,
    build_coverage_sets,
    greedy_coverage_order,
    run_pilot,
)
from .program import ReasoningProgram, compile_reasoning
from .statistics import (
    paired_apfd_deltas,
    permutation_test_apfd_delta,
    source_group_bootstrap_apfd_delta,
)
from .table_program import (
    TableReasoningProgram,
    atomic_reasoning_path_items,
    compile_table_program,
    primary_reasoning_items,
    sequence_reasoning_items,
    z1_category_x_z3_pair_items,
    z3_pair_items,
)

__all__ = [
    "BenchmarkCase",
    "ReasoningState",
    "ReasoningProgram",
    "TableReasoningProgram",
    "atomic_reasoning_path_items",
    "apfd",
    "build_cases",
    "build_coverage_sets",
    "compile_reasoning",
    "compile_table_program",
    "greedy_coverage_order",
    "paired_apfd_deltas",
    "permutation_test_apfd_delta",
    "primary_reasoning_items",
    "run_pilot",
    "sequence_reasoning_items",
    "source_group_bootstrap_apfd_delta",
    "z1_category_x_z3_pair_items",
    "z3_pair_items",
]
