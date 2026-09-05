# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime evidence for the common-root CUTLASS Histogram contract."""

import pytest

from examples.cutlass._group_histogram_codegen_probe import (
    PORTABLE_DTYPE_CASES,
    run_dtype_example,
    run_example,
)

from ..support.runtime import runtime_pytestmark

pytestmark = runtime_pytestmark


@pytest.mark.evidence_for("group.histogram", backend="cutlass", evidence="runtime")
def test_common_histogram_matches_qualified_cutlass_and_independent_oracle() -> None:
    assert run_example() == {
        "algorithms": ("atomic", "sort"),
        "bins": 97,
        "bins_per_thread": 2,
        "block_dim": (8, 4, 2),
        "input_preserved": True,
        "out_of_range_slots_zero": True,
        "repeat_launches": 2,
    }


@pytest.mark.parametrize(
    ("sample_dtype_name", "counter_dtype_name"),
    PORTABLE_DTYPE_CASES,
)
def test_common_histogram_portable_dtype_closure_matches_qualified_cutlass(
    sample_dtype_name: str,
    counter_dtype_name: str,
) -> None:
    assert run_dtype_example(sample_dtype_name, counter_dtype_name) == {
        "algorithms": ("atomic", "sort"),
        "bins": 31,
        "common_qualified_exact": True,
        "counter_dtype": counter_dtype_name,
        "input_preserved": True,
        "out_of_range_slots_zero": True,
        "sample_dtype": sample_dtype_name,
    }
