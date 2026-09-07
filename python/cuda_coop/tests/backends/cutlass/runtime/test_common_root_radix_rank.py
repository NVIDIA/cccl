# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from examples.cutlass._common_root_radix_rank_codegen_probe import run_dtype_example

pytestmark = [pytest.mark.gpu, pytest.mark.runtime]


@pytest.mark.evidence_for("group.radix_rank", backend="cutlass", evidence="runtime")
@pytest.mark.parametrize(
    ("dtype_name", "bit_width", "high_or_sign_bit_interval"),
    [
        ("int32", 32, (24, 32)),
        ("uint32", 32, (24, 32)),
        ("int64", 64, (56, 64)),
        ("uint64", 64, (56, 64)),
    ],
)
def test_common_and_qualified_radix_rank_match_stable_rank_oracles(
    dtype_name: str,
    bit_width: int,
    high_or_sign_bit_interval: tuple[int, int],
) -> None:
    assert run_dtype_example(dtype_name) == {
        "bit_width": bit_width,
        "block_threads": 64,
        "dtype": dtype_name,
        "high_or_sign_bit_interval": high_or_sign_bit_interval,
        "input_preserved": True,
        "items_per_thread": 2,
        "radix_bits_matches_end_bit": True,
        "stable_exact_ranks": True,
    }
