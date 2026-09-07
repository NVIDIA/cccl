# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from examples.cutlass._common_root_merge_sort_codegen_probe import (
    run_dtype_example,
    run_example,
)

pytestmark = [pytest.mark.gpu, pytest.mark.runtime]


@pytest.mark.evidence_for(
    "group.merge_sort_keys", backend="cutlass", evidence="runtime"
)
def test_common_and_qualified_merge_sort_match_independent_oracles() -> None:
    assert run_example() == {
        "block_partial_items": 117,
        "duplicate_keys": True,
        "input_preserved": True,
        "items_per_thread": 2,
        "sentinel_ordering": True,
        "warp_partial_items": 53,
    }


@pytest.mark.evidence_for(
    "group.merge_sort_keys", backend="cutlass", evidence="runtime"
)
@pytest.mark.parametrize("dtype_name", ["uint32", "int64", "uint64"])
def test_portable_integer_dtypes_match_block_and_warp_oracles(
    dtype_name: str,
) -> None:
    assert run_dtype_example(dtype_name) == {
        "dtype": dtype_name,
        "high_bit_or_wide_values": True,
        "input_preserved": True,
        "items_per_thread": 2,
        "physical_warps": 2,
    }
