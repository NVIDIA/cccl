# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_numba_mlir_block_shuffle_adapter_preserves_static_distance_and_boundary():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop._core.block import make_block_shuffle_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter

    scalar = NumbaMlirCoreAdapter().materialize(
        make_block_shuffle_spec(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            mode="offset",
            distance=ArgumentBinding.static(-2),
        ).specialization
    )
    assert scalar.method_name == "Offset"
    assert scalar.fake_return
    assert [type(parameter).__name__ for parameter in scalar.parameters[0]] == [
        "Pointer",
        "Reference",
        "Reference",
        "CxxFunction",
    ]
    assert scalar.parameters[0][2].is_output
    assert scalar.parameters[0][3].cpp == "-2"

    array = NumbaMlirCoreAdapter().materialize(
        make_block_shuffle_spec(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            mode="down",
            items_per_thread=2,
            block_prefix=True,
        ).specialization
    )
    assert array.method_name == "Down"
    assert [type(parameter).__name__ for parameter in array.parameters[0]] == [
        "Pointer",
        "Array",
        "Array",
        "PointerReference",
    ]
    assert not array.parameters[0][2].is_output
    assert not array.parameters[0][3].is_output


def test_numba_mlir_block_shuffle_factory_defines_aggregate_storage(monkeypatch):
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._block import _block_shuffle as module

    monkeypatch.setattr(
        module,
        "make_invocable_from_specialization",
        lambda specialization: specialization,
    )

    specialization = module.shuffle(
        module.BlockShuffleType.Down,
        dtype=types.complex128,
        threads_per_block=32,
        items_per_thread=2,
    )

    assert len(specialization.type_definitions) == 1
    assert "struct __align__(8) storage_t" in specialization.type_definitions[0].code
    assert "char data[16]" in specialization.type_definitions[0].code
