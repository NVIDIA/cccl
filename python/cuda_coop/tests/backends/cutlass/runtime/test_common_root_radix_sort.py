# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from examples.cutlass._common_root_radix_sort_codegen_probe import (
    run_dtype_example,
    run_example,
)

pytestmark = [pytest.mark.gpu, pytest.mark.runtime]


@pytest.mark.evidence_for(
    "group.radix_sort_keys", backend="cutlass", evidence="runtime"
)
def test_common_and_qualified_radix_sort_match_independent_bit_order_oracles() -> None:
    assert run_example() == {
        "begin_only_defaults_to_width": True,
        "block_threads": 64,
        "duplicate_keys": True,
        "input_preserved": True,
        "items_per_thread": 2,
        "signed_bit_order": True,
    }


@pytest.mark.evidence_for(
    "group.radix_sort_keys", backend="cutlass", evidence="runtime"
)
@pytest.mark.parametrize(
    ("dtype_name", "bit_width", "explicit_subrange"),
    [
        ("uint32", 32, (4, 12)),
        ("int64", 64, (36, 52)),
        ("uint64", 64, (36, 52)),
    ],
)
def test_additional_portable_integer_dtypes_match_bit_ordered_oracles(
    dtype_name: str,
    bit_width: int,
    explicit_subrange: tuple[int, int],
) -> None:
    assert run_dtype_example(dtype_name) == {
        "begin_only_defaults_to_width": True,
        "bit_width": bit_width,
        "block_threads": 64,
        "dtype": dtype_name,
        "explicit_subrange": explicit_subrange,
        "high_bit_or_wide_values": True,
        "input_preserved": True,
        "items_per_thread": 2,
    }
