# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported TopK inputs before generating a provider."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _plan(function, arg_types=(), block=(64, 1, 1)):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    planner = _GroupCallPlanner(
        SimpleNamespace(func_ir=run_frontend(function), args=arg_types),
        {"block": block, "grid": (1, 1, 1), "cluster": None},
    )
    return planner.run()


@pytest.mark.parametrize("parameter", ["k", "valid_items"])
@pytest.mark.parametrize("dtype_name", ["boolean", "float32", "uint64"])
def test_runtime_control_types_are_rejected(parameter, dtype_name):
    from numba_cuda_mlir import types

    from cuda import coop

    dtype = getattr(types, dtype_name)
    if parameter == "k":

        def kernel(value):
            keys = coop.ThreadData(2, dtype=types.int32)
            return coop.topk_min_keys(coop.this_block(), keys, k=value)
    else:

        def kernel(value):
            keys = coop.ThreadData(2, dtype=types.int32)
            return coop.topk_min_keys(coop.this_block(), keys, k=1, valid_items=value)

    with pytest.raises(TypeError, match=parameter):
        _plan(kernel, (dtype,))


@pytest.mark.parametrize("dtype_name", ["boolean", "float16", "complex64"])
def test_unsupported_key_types_are_rejected(dtype_name):
    from numba_cuda_mlir import types

    from cuda import coop

    dtype = getattr(types, dtype_name)

    def kernel():
        keys = coop.ThreadData(2, dtype=dtype)
        return coop.topk_max_keys(coop.this_block(), keys, k=1)

    with pytest.raises(TypeError, match="dtype"):
        _plan(kernel)


def test_pair_extents_must_match():
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel():
        keys = coop.ThreadData(2, dtype=types.int32)
        values = coop.ThreadData(3, dtype=types.int32)
        return coop.topk_min_pairs(coop.this_block(), keys, values, k=1)

    with pytest.raises(ValueError, match="matching extents"):
        _plan(kernel)


def test_warp_topk_is_rejected():
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel():
        keys = coop.ThreadData(2, dtype=types.int32)
        return coop.topk_min_keys(coop.this_warp(), keys, k=1)

    with pytest.raises((ValueError, TypeError, NotImplementedError), match="block"):
        _plan(kernel)
