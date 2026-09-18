# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Merge Sort diagnostics before provider compilation."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _plan(function, arg_types=(), block=(64, 1, 1)):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    return _GroupCallPlanner(
        SimpleNamespace(func_ir=run_frontend(function), args=arg_types),
        {"block": block, "grid": (1, 1, 1), "cluster": None},
    ).run()


@pytest.mark.parametrize("count", [-1, 129, 1 << 32, True, 1.5])
def test_invalid_static_counts(count):
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel():
        keys = coop.ThreadData(2, types.int32)
        return coop.merge_sort_keys(
            coop.this_block(), keys, valid_items=count, oob_default=1000
        )

    with pytest.raises((TypeError, ValueError), match="valid_items"):
        _plan(kernel)


@pytest.mark.parametrize("type_name", ["boolean", "float32", "uint64"])
def test_invalid_runtime_counts(type_name):
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel(count):
        keys = coop.ThreadData(2, types.int32)
        return coop.merge_sort_keys(
            coop.this_block(), keys, valid_items=count, oob_default=1000
        )

    with pytest.raises(TypeError, match="valid_items"):
        _plan(kernel, (getattr(types, type_name),))


@pytest.mark.parametrize("default", [None, True, 1.5, 1 << 40])
def test_invalid_sentinel(default):
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel():
        keys = coop.ThreadData(2, types.int32)
        return coop.merge_sort_keys(
            coop.this_block(), keys, valid_items=2, oob_default=default
        )

    with pytest.raises((TypeError, ValueError), match="oob_default"):
        _plan(kernel)


def test_runtime_sentinel_must_match_keys():
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel(sentinel):
        keys = coop.ThreadData(2, types.int32)
        return coop.merge_sort_keys(
            coop.this_block(), keys, valid_items=2, oob_default=sentinel
        )

    with pytest.raises(TypeError, match="oob_default dtype"):
        _plan(kernel, (types.int64,))


def test_pairs_require_equal_extents():
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel():
        keys = coop.ThreadData(2, types.int32)
        values = coop.ThreadData(3, types.float32)
        return coop.merge_sort_pairs(coop.this_block(), keys, values)

    with pytest.raises(ValueError, match="matching.*extents"):
        _plan(kernel)


@pytest.mark.parametrize("descending", [0, "yes"])
def test_descending_requires_static_bool(descending):
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel():
        keys = coop.ThreadData(2, types.int32)
        return coop.merge_sort_keys(coop.this_block(), keys, descending=descending)

    with pytest.raises(TypeError, match="descending.*bool"):
        _plan(kernel)


def test_custom_comparator_controls_order():
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop

    def compare(left, right):
        return left < right

    def kernel():
        keys = numba_coop.ThreadData(2, types.int32)
        return numba_coop.merge_sort_keys(
            numba_coop.this_block(), keys, descending=True, compare_op=compare
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _plan(kernel)


def test_warp_cannot_use_caller_storage():
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel():
        keys = coop.ThreadData(2, types.int32)
        storage = coop.TempStorage()
        return coop.merge_sort_keys(coop.this_warp(), keys, temp_storage=storage)

    with pytest.raises(ValueError, match="only to block"):
        _plan(kernel)


def test_common_rejects_qualified_local_array():
    from numba_cuda_mlir import cuda, types

    from cuda import coop

    def kernel():
        keys = cuda.local.array(2, types.int32)
        return coop.merge_sort_keys(coop.this_block(), keys)

    with pytest.raises(TypeError, match="requires ThreadData"):
        _plan(kernel)
