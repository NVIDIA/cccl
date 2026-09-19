# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Result metadata follows independent output shapes and static variants."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def test_declared_result_dtype_extent_and_arity(monkeypatch):
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda import coop
    from cuda.coop.numba_mlir._compiler import _operations
    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    def marker(group, samples, *, extent=3, dtype=None, mode="one"):
        pass

    result = _operations.GroupResultSource(
        None,
        None,
        fixed_dtype=types.int32,
        dtype_keyword="dtype",
        extent_resolver=lambda context, bound: context.constant(
            bound.arguments["extent"]
        ),
    )

    def results(context, bound):
        return (result,) * (
            2 if context.constant(bound.arguments["mode"]) == "two" else 1
        )

    monkeypatch.setitem(_operations._GROUP_OPERATIONS, marker, "test_result_metadata")
    monkeypatch.setitem(
        _operations._GROUP_PRIMITIVES,
        "test_result_metadata",
        _operations.GroupPrimitiveRegistration(
            lower=lambda *a, **kw: [], result_resolver=results
        ),
    )

    def kernel():
        scalar = types.uint8(1)
        one = marker(coop.this_block(), scalar)
        pair = marker(
            coop.this_block(), scalar, extent=5, dtype=types.uint64, mode="two"
        )
        left, right = pair
        return one, left, right

    func_ir = run_frontend(kernel)
    planner = _GroupCallPlanner(
        SimpleNamespace(func_ir=func_ir, args=()),
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )
    variables = {
        statement.target.name: statement.target
        for block in func_ir.blocks.values()
        for statement in block.body
        if isinstance(statement, ir.Assign)
    }
    for name, extent, dtype in (
        ("one", 3, types.int32),
        ("left", 5, types.uint64),
        ("right", 5, types.uint64),
    ):
        value = variables[name]
        assert planner.context.is_array("test", value)
        assert planner.context.is_thread_data("test", name, value)
        assert planner.context.array_extent(value) == extent
        assert planner.context.dtype(value) == dtype
