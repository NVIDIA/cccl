# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""context.allocate / stackable_context.allocate: raw device memory from a
context's own allocators (cached or uncached); scope-owned inside a scope."""

import numpy as np
import pytest

pytest.importorskip("cuda.stf._experimental._stf_bindings")
rt = pytest.importorskip("cuda.bindings.runtime")
import cuda.stf._experimental as stf  # noqa: E402

NBYTES = 1024


def _round_trip(ctx, ptr):
    """The allocation is asynchronous with respect to the context: complete on
    its fence stream."""
    (err,) = rt.cudaStreamSynchronize(int(ctx.fence()))
    assert err == rt.cudaError_t.cudaSuccess
    src = np.arange(NBYTES, dtype=np.uint8)
    (err,) = rt.cudaMemcpy(
        ptr, src.ctypes.data, NBYTES, rt.cudaMemcpyKind.cudaMemcpyHostToDevice
    )
    assert err == rt.cudaError_t.cudaSuccess
    dst = np.zeros(NBYTES, dtype=np.uint8)
    (err,) = rt.cudaMemcpy(
        dst.ctypes.data, ptr, NBYTES, rt.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )
    assert err == rt.cudaError_t.cudaSuccess
    np.testing.assert_array_equal(dst, src)


@pytest.mark.parametrize("cached", [False, True])
def test_context_allocate(cached):
    with stf.context() as ctx:
        p = ctx.allocate(NBYTES, cached=cached)
        assert isinstance(p, int) and p != 0
        _round_trip(ctx, p)
        ctx.deallocate(p, NBYTES, cached=cached)


def test_context_cached_block_is_recycled():
    with stf.context() as ctx:
        a = ctx.allocate(NBYTES, cached=True)
        ctx.deallocate(a, NBYTES, cached=True)
        b = ctx.allocate(NBYTES, cached=True)
        assert b == a
        ctx.deallocate(b, NBYTES, cached=True)


def test_graph_context_refuses():
    with stf.context(use_graph=True) as ctx:
        with pytest.raises(Exception):
            ctx.allocate(NBYTES)


@pytest.mark.parametrize("cached", [False, True])
def test_stackable_root_allocate(cached):
    with stf.stackable_context() as ctx:
        p = ctx.allocate(NBYTES, cached=cached)
        _round_trip(ctx, p)
        ctx.deallocate(p, NBYTES, cached=cached)


def test_stackable_scope_allocate():
    """Inside a scope: a pointer is handed out while recording, freed before
    the scope ends, and still valid for the scope's graph; the cached allocator
    of the scope serves as well."""
    with stf.stackable_context() as ctx:
        with ctx.graph_scope():
            p = ctx.allocate(NBYTES)
            assert p != 0
            q = ctx.allocate(NBYTES, cached=True)
            assert q != 0 and q != p
            ctx.deallocate(q, NBYTES, cached=True)
            # a task in the scope may touch the memory on its stream
            with ctx.task() as t:
                (err,) = rt.cudaMemsetAsync(p, 7, NBYTES, int(t.stream_ptr()))
                assert err == rt.cudaError_t.cudaSuccess
            ctx.deallocate(p, NBYTES)
