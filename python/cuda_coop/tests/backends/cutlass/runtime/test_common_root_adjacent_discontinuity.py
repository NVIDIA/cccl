# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime evidence for common-root CUTLASS comparison collectives."""

import pytest

from examples.cutlass._group_adjacent_discontinuity_codegen_probe import run_example

from ..support.runtime import runtime_pytestmark

pytestmark = runtime_pytestmark


@pytest.mark.evidence_for(
    "group.adjacent_difference", backend="cutlass", evidence="runtime"
)
@pytest.mark.evidence_for("group.discontinuity", backend="cutlass", evidence="runtime")
def test_common_comparison_cohort_matches_qualified_cutlass_and_oracle() -> None:
    assert run_example() == {
        "block_threads": 64,
        "items_per_thread": 2,
        "valid_items": 125,
        "input_preserved": True,
        "flag_dtype": "int32",
    }
