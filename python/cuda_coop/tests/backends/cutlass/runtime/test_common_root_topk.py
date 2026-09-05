# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from examples.cutlass._common_root_topk_codegen_probe import run_dtype_example

pytestmark = [pytest.mark.gpu, pytest.mark.runtime]


@pytest.mark.evidence_for("group.topk_max_keys", backend="cutlass", evidence="runtime")
@pytest.mark.evidence_for("group.topk_min_keys", backend="cutlass", evidence="runtime")
@pytest.mark.parametrize("dtype_name", ["int32", "uint32", "int64", "uint64"])
def test_common_and_qualified_topk_match_independent_selection_oracles(
    dtype_name: str,
) -> None:
    assert run_dtype_example(dtype_name) == {
        "block_threads": 64,
        "dtype": dtype_name,
        "duplicate_keys": True,
        "full_and_partial": True,
        "high_bit_values": True,
        "input_preserved": True,
        "items_per_thread": 2,
        "runtime_controls": True,
    }
