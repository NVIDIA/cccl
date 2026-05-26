# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

import cuda.stf._experimental as stf


def test_ctx():
    with stf.context():
        pass


def test_graph_ctx():
    ctx = stf.context(use_graph=True)
    ctx.finalize()


def test_ctx2():
    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)
    Z = np.ones(16, dtype=np.float32)

    ctx = stf.context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    t = ctx.task(lX.rw())
    t.start()
    t.end()

    t2 = ctx.task(lX.read(), lY.rw())
    t2.start()
    t2.end()

    t3 = ctx.task(lX.read(), lZ.rw())
    t3.start()
    t3.end()

    t4 = ctx.task(lY.read(), lZ.rw())
    t4.start()
    t4.end()

    ctx.finalize()


def test_ctx3():
    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)
    Z = np.ones(16, dtype=np.float32)

    ctx = stf.context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    with ctx.task(lX.rw()):
        pass

    with ctx.task(lX.read(), lY.rw()):
        pass

    with ctx.task(lX.read(), lZ.rw()):
        pass

    with ctx.task(lY.read(), lZ.rw()):
        pass

    ctx.finalize()


def test_task_arg_cai_v3():
    X = np.ones(16, dtype=np.float32)

    ctx = stf.context()
    lX = ctx.logical_data(X)

    with ctx.task(lX.read()) as t:
        cai = t.get_arg_cai(0).__cuda_array_interface__
        assert cai["version"] == 3
        assert cai["shape"] == X.shape
        assert cai["typestr"] == X.dtype.str
        assert cai["stream"] == t.stream_ptr()

    ctx.finalize()


def test_logical_data_rejects_non_contiguous():
    arr = np.ones((10, 10), dtype=np.float32)
    strided_view = arr[
        ::2, :
    ]  # non-contiguous: stride along axis 0 != itemsize * shape[1]
    assert not strided_view.flags["C_CONTIGUOUS"]

    ctx = stf.context()
    with pytest.raises(ValueError, match="not contiguous"):
        ctx.logical_data(strided_view)
    ctx.finalize()


def test_fence_returns_stream():
    """fence() returns a non-zero CUDA stream handle."""
    ctx = stf.context()
    ld = ctx.logical_data(np.zeros(8, dtype=np.float32))
    with ctx.task(ld.rw()):
        pass
    stream = ctx.fence()
    assert isinstance(stream, int)
    assert stream != 0, "fence() should return a valid (non-zero) CUDA stream"
    ctx.finalize()


def test_fence_graph_ctx():
    """fence() works with a graph-mode context."""
    ctx = stf.context(use_graph=True)
    ld = ctx.logical_data(np.ones(4, dtype=np.float64))
    with ctx.task(ld.rw()):
        pass
    stream = ctx.fence()
    assert isinstance(stream, int)
    assert stream != 0
    ctx.finalize()


def test_fence_then_more_tasks():
    """Tasks can be submitted after fence()."""
    ctx = stf.context()
    arr = np.zeros(4, dtype=np.float32)
    ld = ctx.logical_data(arr)

    with ctx.task(ld.rw()):
        pass

    stream1 = ctx.fence()
    assert stream1 != 0

    with ctx.task(ld.rw()):
        pass

    stream2 = ctx.fence()
    assert stream2 != 0

    ctx.finalize()


def test_fence_multiple_deps():
    """fence() works with multiple logical data in flight."""
    ctx = stf.context()
    X = np.ones(8, dtype=np.float32)
    Y = np.ones(8, dtype=np.float32)
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)

    with ctx.task(lX.read(), lY.rw()):
        pass

    stream = ctx.fence()
    assert isinstance(stream, int)
    assert stream != 0
    ctx.finalize()


def test_fence_on_null_ctx_raises():
    """fence() raises RuntimeError on an already-finalized context."""
    ctx = stf.context()
    ctx.finalize()
    with pytest.raises(RuntimeError, match="context handle is NULL"):
        ctx.fence()


def test_double_finalize_is_safe():
    """Calling finalize() twice is a no-op: the Python guard NULLs the
    handle before calling C++, so the second call never reaches the C API."""
    ctx = stf.context()
    ctx.finalize()
    ctx.finalize()


def test_borrowed_context_cannot_finalize():
    """A borrowed context must raise on finalize()."""
    ctx = stf.context()
    ld = ctx.logical_data(np.zeros(4, dtype=np.float32))
    borrowed = ld.borrow_ctx_handle()
    with pytest.raises(RuntimeError, match="cannot finalize borrowed context"):
        borrowed.finalize()
    ctx.finalize()


def test_finalize_then_fence_raises():
    """Operations on a finalized context raise RuntimeError."""
    ctx = stf.context()
    ctx.finalize()
    with pytest.raises(RuntimeError, match="context handle is NULL"):
        ctx.fence()


def test_dealloc_does_not_raise():
    """Deleting already-finalized objects should never raise."""
    ctx = stf.context()
    ld = ctx.logical_data(np.zeros(4, dtype=np.float32))
    ctx.finalize()
    del ld
    del ctx


def test_context_manager_finalizes():
    with stf.context() as ctx:
        ld = ctx.logical_data(np.zeros(4, dtype=np.float32))
        with ctx.task(ld.rw()):
            pass

    with pytest.raises(RuntimeError, match="context handle is NULL"):
        ctx.fence()


def test_unfinalized_context_warns():
    with pytest.warns(ResourceWarning, match="without an explicit finalize"):
        ctx = stf.context()
        del ctx


if __name__ == "__main__":
    test_ctx3()
