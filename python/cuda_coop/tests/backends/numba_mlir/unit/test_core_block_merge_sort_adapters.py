# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_numba_mlir_block_merge_sort_adapter_preserves_payload_and_partial_abi():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core import CxxOperator, Dependency
    from cuda.coop._core.block import make_block_merge_sort_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._types import algo_coalesce_key

    partial = make_block_merge_sort_spec(
        key_dtype=types.int32,
        value_dtype=types.float32,
        block_dim=(16, 2, 1),
        items_per_thread=2,
        compare_operator=CxxOperator(
            "::cuda::std::less<KeyT>",
            Dependency("KeyT"),
            name="compare_op",
        ),
        valid_items=31,
        oob_default=0,
    )
    partial_specialization = NumbaMlirCoreAdapter().materialize(partial.specialization)
    assert [
        type(parameter).__name__ for parameter in partial_specialization.parameters[0]
    ] == ["Pointer", "Array", "Array", "CxxFunction", "Value", "Reference"]
    assert not partial_specialization.parameters[0][1].is_output
    assert not partial_specialization.parameters[0][2].is_output

    full = make_block_merge_sort_spec(
        key_dtype=types.int32,
        block_dim=(16, 2, 1),
        items_per_thread=2,
        compare_operator=CxxOperator(
            "::cuda::std::less<KeyT>",
            Dependency("KeyT"),
            name="compare_op",
        ),
    )
    full_specialization = NumbaMlirCoreAdapter().materialize(full.specialization)
    assert algo_coalesce_key(partial_specialization) != algo_coalesce_key(
        full_specialization
    )
