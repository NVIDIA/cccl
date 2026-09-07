# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_numba_mlir_block_radix_sort_adapter_preserves_both_overloads():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core.block import (
        BlockRadixSortBitPolicy,
        make_block_radix_sort_spec,
    )
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter

    core_spec = make_block_radix_sort_spec(
        key_dtype=types.uint32,
        value_dtype=types.float32,
        block_dim=(32, 1, 1),
        items_per_thread=2,
        bit_policy=BlockRadixSortBitPolicy.BOTH,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)

    assert [
        [type(parameter).__name__ for parameter in method]
        for method in specialization.parameters
    ] == [
        ["Pointer", "Array", "Array"],
        ["Pointer", "Array", "Array", "Value", "Value"],
    ]
    assert not specialization.parameters[0][1].is_output
    assert not specialization.parameters[0][2].is_output
