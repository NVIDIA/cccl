# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported neighbor arguments before provider compilation."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _plan(function, arg_types=()):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    return _GroupCallPlanner(
        SimpleNamespace(func_ir=run_frontend(function), args=arg_types),
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()


@pytest.mark.parametrize("type_name", ["boolean", "float32", "uint64"])
def test_runtime_count_requires_supported_integer(type_name):
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel(count):
        values = coop.ThreadData(2, types.int32)
        return coop.adjacent_difference(coop.this_block(), values, valid_items=count)

    with pytest.raises(TypeError, match="valid_items"):
        _plan(kernel, (getattr(types, type_name),))


@pytest.mark.parametrize("operation", ["adjacent_difference", "discontinuity"])
def test_runtime_boundary_dtype_must_match(operation):
    from numba_cuda_mlir import types

    from cuda import coop

    function = getattr(coop, operation)

    def kernel(boundary):
        values = coop.ThreadData(2, types.float32)
        return function(coop.this_block(), values, tile_predecessor_item=boundary)

    with pytest.raises(TypeError, match="tile_predecessor_item dtype"):
        _plan(kernel, (types.float64,))


@pytest.mark.parametrize("operation", ["adjacent_difference", "discontinuity"])
def test_reject_scalar_and_warp_inputs(operation):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop

    function = getattr(numba_coop, operation)

    def scalar():
        return function(numba_coop.this_block(), types.int32(1))

    with pytest.raises(TypeError, match="fixed-size"):
        _plan(scalar)

    def warp():
        values = numba_coop.ThreadData(2, types.int32)
        return function(numba_coop.this_warp(), values)

    with pytest.raises(Exception, match="complete block"):
        _plan(warp)


def test_mode_requests_literal_specialization():
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel(runtime_mode):
        values = coop.ThreadData(2, types.int32)
        return coop.discontinuity(coop.this_block(), values, mode=runtime_mode)

    from numba_cuda_mlir.numba_cuda.core.errors import ForceLiteralArg

    # The dispatcher must specialize the selector before choosing output arity.
    with pytest.raises(ForceLiteralArg):
        _plan(kernel, (types.unicode_type,))


@pytest.mark.parametrize("operation", ["adjacent_difference", "discontinuity"])
def test_neighbor_operator_tracks_nested_device_helper(monkeypatch, operation):
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir.descriptor import MLIRDispatcher

    from cuda.coop._core import _symbols, semantic_token
    from cuda.coop.numba_mlir._lowering._neighbors import neighbor_operator

    original = _symbols._type_dependency_token

    def reject_dispatcher_class(value, state):
        assert value is not MLIRDispatcher, "fingerprinting compiler implementation"
        return original(value, state)

    monkeypatch.setattr(_symbols, "_type_dependency_token", reject_dispatcher_class)

    def make_operator(offset):
        @cuda.jit(device=True)
        def helper(left, right):
            return left - right + offset

        def callback(left, right):
            return helper(left, right)

        return neighbor_operator(operation, callback)

    first = semantic_token(make_operator(1))
    assert first == semantic_token(make_operator(1))
    assert first != semantic_token(make_operator(2))
