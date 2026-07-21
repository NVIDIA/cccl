# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402


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
        # The view advertises no stream: STF already orders the task stream
        # behind the data's producers, and reporting an integer stream would
        # make consumers such as Numba host-synchronize (illegal during graph
        # capture). Callers launch their work on t.stream_ptr() directly.
        assert cai["stream"] is None

    ctx.finalize()


@pytest.mark.parametrize("context_type", [stf.context, stf.stackable_context])
@pytest.mark.parametrize(
    "bad_shape, match",
    [
        pytest.param((), "at least one dimension", id="empty"),
        pytest.param((0,), "positive", id="zero"),
        pytest.param((4, 0), "positive", id="zero-second-axis"),
        pytest.param((-1,), "positive", id="negative"),
        pytest.param((2.5,), "integers", id="non-integral"),
    ],
)
def test_logical_data_empty_rejects_invalid_shape(context_type, bad_shape, match):
    ctx = context_type()
    with pytest.raises((ValueError, TypeError), match=match):
        ctx.logical_data_empty(bad_shape, dtype=np.float32)
    ctx.finalize()


def test_logical_data_rejects_non_contiguous():
    arr = np.ones((10, 10), dtype=np.float32)
    strided_view = arr[
        ::2, :
    ]  # non-contiguous: stride along axis 0 != itemsize * shape[1]
    assert not strided_view.flags["C_CONTIGUOUS"]

    ctx = stf.context()
    with pytest.raises(ValueError, match="C-contiguous"):
        ctx.logical_data(strided_view)
    ctx.finalize()


class _CudaArrayInterfaceWrapper:
    def __init__(self, array):
        self._array = array
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": array.shape,
            "typestr": array.dtype.str,
            "data": (array.ctypes.data, False),
            "strides": array.strides,
        }


@pytest.mark.parametrize("context_type", [stf.context, stf.stackable_context])
@pytest.mark.parametrize(
    "make_view",
    [
        pytest.param(lambda array: array[::2], id="strided"),
        pytest.param(lambda array: array[::-1], id="negative-stride"),
        pytest.param(lambda array: array.reshape(2, 4).T, id="transposed"),
    ],
)
def test_logical_data_rejects_non_contiguous_cai(context_type, make_view):
    view = make_view(np.arange(8, dtype=np.float32))
    assert not view.flags["C_CONTIGUOUS"]

    ctx = context_type()
    with pytest.raises(ValueError, match="not C-contiguous"):
        ctx.logical_data(_CudaArrayInterfaceWrapper(view))
    ctx.finalize()


@pytest.mark.parametrize("context_type", [stf.context, stf.stackable_context])
def test_logical_data_accepts_explicit_c_contiguous_cai_strides(context_type):
    array = np.arange(8, dtype=np.float32).reshape(2, 4)
    assert array.strides is not None

    ctx = context_type()
    ld = ctx.logical_data(_CudaArrayInterfaceWrapper(array))
    assert ld.shape == array.shape
    ctx.finalize()


class _ReadonlyCudaArrayInterfaceWrapper:
    def __init__(self, array):
        self._array = array
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": array.shape,
            "typestr": array.dtype.str,
            "data": (array.ctypes.data, True),  # readonly export
            "strides": None,
        }


@pytest.mark.parametrize("context_type", [stf.context, stf.stackable_context])
def test_logical_data_readonly_buffer_rejects_write_deps(context_type):
    array = np.arange(8, dtype=np.float64)
    array.setflags(write=False)

    ctx = context_type()
    ld = ctx.logical_data(array)
    assert ld.readonly
    ld.read()  # read access stays legal
    with pytest.raises(ValueError, match="read-only"):
        ld.write()
    with pytest.raises(ValueError, match="read-only"):
        ld.rw()
    ctx.finalize()


@pytest.mark.parametrize("context_type", [stf.context, stf.stackable_context])
def test_logical_data_readonly_cai_rejects_write_deps(context_type):
    array = np.arange(8, dtype=np.float64)

    ctx = context_type()
    ld = ctx.logical_data(_ReadonlyCudaArrayInterfaceWrapper(array))
    assert ld.readonly
    ld.read()
    with pytest.raises(ValueError, match="read-only"):
        ld.write()
    with pytest.raises(ValueError, match="read-only"):
        ld.rw()
    ctx.finalize()


@pytest.mark.parametrize("context_type", [stf.context, stf.stackable_context])
def test_logical_data_writable_source_not_readonly(context_type):
    array = np.zeros(8, dtype=np.float64)

    ctx = context_type()
    ld = ctx.logical_data(array)
    assert not ld.readonly
    ld.write()
    ld.rw()
    ctx.finalize()


def test_stackable_readonly_source_safe_across_scopes():
    # A read-only source is auto-marked STF read-only, so nested scopes
    # auto-import it with READ (no RW freeze, no write-back into the
    # immutable source) and write-capable explicit pushes are rejected.
    array = np.arange(8, dtype=np.float64)
    array.setflags(write=False)

    ctx = stf.stackable_context()
    ld = ctx.logical_data(array)
    assert ld.readonly
    with pytest.raises(ValueError, match="read-only"):
        ld.push(stf.AccessMode.RW)
    with ctx.graph_scope():
        ld.push(stf.AccessMode.READ)
    ctx.finalize()


def test_stackable_set_read_only_blocks_write_deps():
    array = np.zeros(8, dtype=np.float64)

    ctx = stf.stackable_context()
    ld = ctx.logical_data(array)
    assert not ld.readonly
    ld.set_read_only()
    assert ld.readonly
    with pytest.raises(ValueError, match="read-only"):
        ld.write()
    with pytest.raises(ValueError, match="read-only"):
        ld.push(stf.AccessMode.RW)
    ctx.finalize()


@pytest.mark.parametrize("context_type", [stf.context, stf.stackable_context])
def test_logical_data_pins_buffer_protocol_export(context_type):
    # The Py_buffer export must stay active for the logical data's lifetime:
    # STF holds the raw pointer, so a resizable exporter (bytearray) must be
    # blocked from reallocating out from under it.
    buf = bytearray(64)

    ctx = context_type()
    ld = ctx.logical_data(buf)
    with pytest.raises(BufferError):
        buf.extend(b"x")  # resize attempt while the export is held
    del ld
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
        # Force destruction while pytest is still checking for the warning.
        gc.collect()
