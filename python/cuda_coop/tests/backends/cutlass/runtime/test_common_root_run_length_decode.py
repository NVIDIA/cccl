# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from examples.cutlass._common_root_run_length_decode_codegen_probe import (
    PORTABLE_DTYPE_CASES,
    run_dtype_example,
)

pytestmark = [pytest.mark.gpu, pytest.mark.runtime]


@pytest.mark.evidence_for(
    "group.run_length_decode",
    backend="cutlass",
    evidence="runtime",
)
@pytest.mark.parametrize(
    ("value_dtype_name", "length_dtype_name"),
    PORTABLE_DTYPE_CASES,
)
def test_common_and_qualified_decode_match_partial_and_oob_window_oracles(
    value_dtype_name: str,
    length_dtype_name: str,
) -> None:
    length_bits = 64 if length_dtype_name.endswith("64") else 32
    offset_bits = length_bits if length_dtype_name.startswith("u") else length_bits - 1
    if length_dtype_name in {"int64", "uint64"}:
        decoded_total = (1 << offset_bits) - 1
        partial_valid_items = 5
    else:
        decoded_total = sum(1 + (index * 7) % 4 for index in range(96))
        partial_valid_items = 17
    relative_oob_sentinel = (
        -1 if not length_dtype_name.startswith("u") else (1 << length_bits) - 1
    )
    assert run_dtype_example(value_dtype_name, length_dtype_name) == {
        "after_total_zero_filled": True,
        "block_dim": (8, 4, 2),
        "common_qualified_exact": True,
        "decoded_items_per_thread": 3,
        "decoded_total": decoded_total,
        "genuine_64bit_window": length_dtype_name.endswith("64"),
        "input_preserved": True,
        "length_dtype": length_dtype_name,
        "maximum_offset": (1 << offset_bits) - 1,
        "maximum_offset_zero_filled": True,
        "multi_run": True,
        "partial_tail_zero_filled": True,
        "partial_valid_items": partial_valid_items,
        "relative_oob_sentinel": relative_oob_sentinel,
        "repeat_launches": 2,
        "runs_per_thread": 2,
        "value_dtype": value_dtype_name,
        "trailing_zero_length_padding": True,
    }
