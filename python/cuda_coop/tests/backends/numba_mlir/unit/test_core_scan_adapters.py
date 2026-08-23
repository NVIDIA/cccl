# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def _multiply(left, right):
    return left * right


class _RunningPrefix:
    def __call__(self_ptr, aggregate):
        old_prefix = self_ptr[0]
        self_ptr[0] = old_prefix + aggregate
        return old_prefix


def test_numba_mlir_scan_adapter_lowers_stateless_and_stateful_operators():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core import (
        CxxOperator,
        Dependency,
        PythonOperator,
        StatefulOperator,
    )
    from cuda.coop._core.block import make_block_scan_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._types import StatefulFunction

    prefix = StatefulFunction(
        _RunningPrefix,
        types.int32,
        name="core_scan_running_prefix",
    )
    core_spec = make_block_scan_spec(
        dtype=types.int32,
        block_dim=(32, 1, 1),
        items_per_thread=2,
        mode="exclusive",
        algorithm="::cub::BLOCK_SCAN_RAKING",
        value_kind="array",
        scan_operator=PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=_multiply,
            name="scan_op",
        ),
        prefix_operator=StatefulOperator(
            op=prefix,
            state_dtype=types.int32,
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"),),
            name="prefix_op",
        ),
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)

    assert [type(parameter).__name__ for parameter in specialization.parameters[0]][
        -2:
    ] == ["StatelessOperator", "StatefulOperator"]

    cxx_spec = make_block_scan_spec(
        dtype=types.int32,
        block_dim=(32, 1, 1),
        items_per_thread=1,
        mode="inclusive",
        algorithm="::cub::BLOCK_SCAN_RAKING",
        value_kind="array",
        scan_operator=CxxOperator(
            cpp="::cuda::std::multiplies<T>",
            dtype=Dependency("T"),
            name="scan_op",
        ),
    )
    cxx_specialization = NumbaMlirCoreAdapter().materialize(cxx_spec.specialization)
    assert type(cxx_specialization.parameters[0][-1]).__name__ == "CxxFunction"
