# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_numba_mlir_histogram_normalizes_python_int_locally():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._block._block_histogram import (
        _normalize_histogram_dtype,
    )

    assert _normalize_histogram_dtype(int) == types.int32
    assert _normalize_histogram_dtype(types.uint8) == types.uint8


def test_numba_mlir_histogram_adapter_preserves_init_and_composite_abis():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core.block import make_block_histogram_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._types import algo_coalesce_key

    init = make_block_histogram_spec(
        item_dtype=types.uint8,
        counter_dtype=types.uint32,
        block_dim=(16, 2, 1),
        items_per_thread=4,
        bins=256,
        algorithm="atomic",
        operation="init",
    )
    init_specialization = NumbaMlirCoreAdapter().materialize(init.specialization)
    assert [
        type(parameter).__name__ for parameter in init_specialization.parameters[0]
    ] == ["Pointer", "Array"]

    composite = make_block_histogram_spec(
        item_dtype=types.uint8,
        counter_dtype=types.uint32,
        block_dim=(16, 2, 1),
        items_per_thread=4,
        bins=256,
        algorithm="atomic",
        operation="composite",
    )
    composite_specialization = NumbaMlirCoreAdapter().materialize(
        composite.specialization
    )
    assert [
        type(parameter).__name__ for parameter in composite_specialization.parameters[0]
    ] == ["Pointer", "Array", "Array"]
    assert not composite_specialization.parameters[0][2].is_output
    assert algo_coalesce_key(init_specialization) != algo_coalesce_key(
        composite_specialization
    )
