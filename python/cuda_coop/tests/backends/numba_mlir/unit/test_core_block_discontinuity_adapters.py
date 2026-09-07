# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def _not_equal(left, right):
    return left != right


def _complex_not_equal(left, right):
    return left.real != right.real or left.imag != right.imag


def test_numba_mlir_adapter_preserves_boundary_overload_abi_and_identity():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core import Dependency, PythonOperator
    from cuda.coop._core.block import make_block_discontinuity_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._types import algo_coalesce_key

    def make(*, predecessor=None, successor=None):
        return make_block_discontinuity_spec(
            dtype=types.int32,
            flag_dtype=types.boolean,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            mode="heads_and_tails",
            flag_operator=PythonOperator(
                Dependency("FlagT"),
                (Dependency("T"), Dependency("T")),
                _not_equal,
                name="flag_op",
            ),
            tile_predecessor_item=predecessor,
            tile_successor_item=successor,
        )

    paired = make(predecessor=0, successor=9)
    paired_specialization = NumbaMlirCoreAdapter().materialize(paired.specialization)
    assert [
        type(parameter).__name__ for parameter in paired_specialization.parameters[0]
    ] == [
        "Pointer",
        "Array",
        "Reference",
        "Array",
        "Reference",
        "Array",
        "StatelessOperator",
    ]
    assert not paired_specialization.parameters[0][1].is_output
    assert not paired_specialization.parameters[0][3].is_output

    plain_specialization = NumbaMlirCoreAdapter().materialize(make().specialization)
    assert algo_coalesce_key(paired_specialization) != algo_coalesce_key(
        plain_specialization
    )


def test_numba_mlir_discontinuity_factory_defines_aggregate_storage(monkeypatch):
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._block import _block_discontinuity as module

    monkeypatch.setattr(
        module,
        "make_invocable_from_specialization",
        lambda specialization: specialization,
    )

    specialization = module.discontinuity(
        dtype=types.complex128,
        threads_per_block=32,
        items_per_thread=2,
        flag_op=_complex_not_equal,
        flag_dtype=types.int32,
    )

    assert len(specialization.type_definitions) == 1
    assert "struct __align__(8) storage_t" in specialization.type_definitions[0].code
    assert "char data[16]" in specialization.type_definitions[0].code
