# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def _subtract(left, right):
    return left - right


def test_numba_mlir_adapter_preserves_full_and_partial_overload_abis():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core import Dependency, PythonOperator
    from cuda.coop._core.block import make_block_adjacent_difference_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._types import algo_coalesce_key

    full = make_block_adjacent_difference_spec(
        dtype=types.int32,
        block_dim=(32, 1, 1),
        items_per_thread=2,
        direction="right",
        difference_operator=PythonOperator(
            Dependency("T"),
            (Dependency("T"), Dependency("T")),
            _subtract,
            name="difference_op",
        ),
        tile_successor_item=0,
    )
    full_specialization = NumbaMlirCoreAdapter().materialize(full.specialization)
    assert [
        type(parameter).__name__ for parameter in full_specialization.parameters[0]
    ] == ["Pointer", "Array", "Array", "StatelessOperator", "Reference"]
    assert not full_specialization.parameters[0][2].is_output

    partial = make_block_adjacent_difference_spec(
        dtype=types.int32,
        block_dim=(32, 1, 1),
        items_per_thread=2,
        direction="left",
        difference_operator=PythonOperator(
            Dependency("T"),
            (Dependency("T"), Dependency("T")),
            _subtract,
            name="difference_op",
        ),
        valid_items=31,
        tile_predecessor_item=0,
    )
    partial_specialization = NumbaMlirCoreAdapter().materialize(partial.specialization)
    assert [
        type(parameter).__name__ for parameter in partial_specialization.parameters[0]
    ] == [
        "Pointer",
        "Array",
        "Array",
        "StatelessOperator",
        "Value",
        "Reference",
    ]
    assert algo_coalesce_key(full_specialization) != algo_coalesce_key(
        partial_specialization
    )


def test_numba_mlir_adjacent_difference_factory_defines_aggregate_storage(
    monkeypatch,
):
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._block import _block_adjacent_difference as module

    monkeypatch.setattr(
        module,
        "make_invocable_from_specialization",
        lambda specialization: specialization,
    )

    specialization = module.adjacent_difference(
        module.BlockAdjacentDifferenceType.SubtractLeft,
        dtype=types.complex128,
        threads_per_block=32,
        items_per_thread=2,
        difference_op=_subtract,
    )

    assert len(specialization.type_definitions) == 1
    assert "struct __align__(8) storage_t" in specialization.type_definitions[0].code
    assert "char data[16]" in specialization.type_definitions[0].code
