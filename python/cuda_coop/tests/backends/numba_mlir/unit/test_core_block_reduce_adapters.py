# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def _add(left, right):
    return left + right


def test_numba_mlir_block_reduce_adapter_preserves_scalar_array_and_operator_abi():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core import Dependency, PythonOperator
    from cuda.coop._core.block import make_block_reduce_spec
    from cuda.coop.numba_mlir._block._block_reduce import sum as block_sum
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter

    with pytest.raises(ValueError, match="num_valid is not supported for array"):
        block_sum(
            types.int32,
            threads_per_block=32,
            items_per_thread=2,
            num_valid=17,
        )

    scalar = make_block_reduce_spec(
        dtype=types.int32,
        block_dim=(32, 1, 1),
        items_per_thread=1,
        operation="reduce",
        algorithm="::cub::BLOCK_REDUCE_WARP_REDUCTIONS",
        value_kind="scalar",
        reduce_operator=PythonOperator(
            Dependency("T"),
            (Dependency("T"), Dependency("T")),
            _add,
            name="binary_op",
        ),
        valid_items=True,
    )
    scalar_specialization = NumbaMlirCoreAdapter().materialize(scalar.specialization)
    assert [
        type(parameter).__name__ for parameter in scalar_specialization.parameters[0]
    ] == ["Pointer", "Reference", "StatelessOperator", "Value", "Reference"]
    assert scalar_specialization.parameters[0][-1].is_output
    assert scalar_specialization._symbol_base_name().endswith("_Reduce")

    array = make_block_reduce_spec(
        dtype=types.int32,
        block_dim=(32, 1, 1),
        items_per_thread=2,
        operation="sum",
        algorithm="::cub::BLOCK_REDUCE_RAKING",
        value_kind="array",
    )
    array_specialization = NumbaMlirCoreAdapter().materialize(array.specialization)
    assert [
        type(parameter).__name__ for parameter in array_specialization.parameters[0]
    ] == ["Pointer", "Array", "Reference"]
