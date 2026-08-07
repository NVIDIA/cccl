# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""DLPack producer tests for :class:`DeviceArray`.

DLPack is the OWNERSHIP-CARRYING companion to the CUDA Array Interface:
``torch.from_dlpack(arr)`` yields a tensor whose storage keeps the
allocation alive, freed by the ``DeviceArray`` finalizer once the last
consumer dies. CAI remains the borrowed / zero-copy path. Both protocols
coexist on every array; a consumer picks by construction.
"""

import gc
import weakref

import numpy as np
import pytest

pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402

torch = pytest.importorskip("torch")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


@pytest.fixture()
def dplace():
    stf.machine_init()
    return stf.data_place.device(0)


def _filled(shape, dtype, dplace):
    host = (
        np.arange(np.prod(shape)).reshape(shape).astype(dtype)
        if np.dtype(dtype).kind != "b"
        else (np.arange(np.prod(shape)).reshape(shape) % 2).astype(bool)
    )
    arr = stf.DeviceArray(int(np.prod(shape)), dtype, dplace)
    arr.copy_to_device(host)
    return arr, host


# -- protocol surface ---------------------------------------------------------


@requires_cuda
def test_dlpack_device(dplace):
    arr = stf.DeviceArray(32, np.float32, dplace)
    assert arr.__dlpack_device__() == (2, 0)  # (kDLCUDA, ordinal)


@requires_cuda
def test_capsule_is_consumable_dltensor(dplace):
    """An explicit capsule is a valid "dltensor" and can be consumed once."""
    arr, host = _filled((15,), np.float32, dplace)
    cap = arr.__dlpack__()
    assert "dltensor" in repr(cap)
    t = torch.from_dlpack(cap)  # torch accepts a raw capsule
    assert np.array_equal(t.cpu().numpy(), host)


@requires_cuda
def test_max_version_and_kwargs_accepted(dplace):
    """``max_version``/``dl_device``/``copy`` follow the DLPack 1.x calling
    convention; the producer answers with an unversioned capsule."""
    arr, host = _filled((4,), np.float32, dplace)
    cap = arr.__dlpack__(max_version=(1, 1), dl_device=(2, 0), copy=False)
    assert np.array_equal(torch.from_dlpack(cap).cpu().numpy(), host)


# -- round trips --------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize(
    "np_dtype", [np.float32, np.float64, np.int32, np.int64, np.uint8, np.bool_]
)
def test_from_dlpack_roundtrip_dtypes(dplace, np_dtype):
    arr, host = _filled((42,), np_dtype, dplace)
    t = torch.from_dlpack(arr)
    assert tuple(t.shape) == (42,)
    assert t.device.type == "cuda"
    assert np.array_equal(t.cpu().numpy(), host)


@requires_cuda
def test_writes_visible_both_ways(dplace):
    """DLPack import is zero-copy: writes through the tensor are visible via
    the DeviceArray (and its CAI view), and vice versa."""
    arr, host = _filled((64,), np.float32, dplace)
    t = torch.from_dlpack(arr)
    t += 1.0
    torch.cuda.synchronize()
    assert np.array_equal(arr.copy_to_host(), host + 1.0)
    arr.copy_to_device(host * 2.0)
    assert np.array_equal(t.cpu().numpy(), host * 2.0)


@requires_cuda
def test_zero_size_export(dplace):
    arr = stf.DeviceArray(0, np.float32, dplace)
    t = torch.from_dlpack(arr)
    assert t.numel() == 0 and t.dtype == torch.float32


# -- ownership ----------------------------------------------------------------


@requires_cuda
def test_tensor_carries_the_allocation(dplace):
    """The imported tensor's storage owns the buffer: dropping every direct
    reference keeps the memory alive; dropping the tensor frees it."""
    arr, host = _filled((256,), np.float32, dplace)
    ref = weakref.ref(arr)
    fin = arr._finalizer_ref
    t = torch.from_dlpack(arr)
    del arr
    gc.collect()
    assert ref() is not None and fin.alive  # capsule keeps the owner alive
    assert np.array_equal(t.cpu().numpy(), host)  # and the data is intact
    del t
    gc.collect()
    assert ref() is None and not fin.alive  # single deallocation point ran


@requires_cuda
def test_multiple_exports_share_one_owner(dplace):
    """Each export holds its own owner reference; the buffer dies only after
    the LAST consumer does."""
    arr, host = _filled((16,), np.float32, dplace)
    ref = weakref.ref(arr)
    t1 = torch.from_dlpack(arr)
    t2 = torch.from_dlpack(arr)
    del arr
    del t1
    gc.collect()
    assert ref() is not None
    assert np.array_equal(t2.cpu().numpy(), host)
    del t2
    gc.collect()
    assert ref() is None


@requires_cuda
def test_unconsumed_capsule_does_not_leak(dplace):
    """A capsule nobody imports runs the deleter from the capsule destructor
    (the "dltensor"/"used_dltensor" rename contract)."""
    arr = stf.DeviceArray(4, np.float32, dplace)
    ref = weakref.ref(arr)
    cap = arr.__dlpack__()
    del arr
    gc.collect()
    assert ref() is not None  # held by the unconsumed capsule
    del cap
    gc.collect()
    assert ref() is None  # destructor released the owner


@requires_cuda
def test_view_export_keeps_root_alive(dplace):
    """Exporting a reshape/slice view transfers ownership of the ROOT
    allocation; view geometry is what the consumer sees."""
    arr, host = _filled((32,), np.float32, dplace)
    root_ref = weakref.ref(arr)
    t = torch.from_dlpack(arr[8:24])
    del arr
    gc.collect()
    assert root_ref() is not None
    assert np.array_equal(t.cpu().numpy(), host[8:24])
    del t
    gc.collect()
    assert root_ref() is None


# -- coexistence with CAI -----------------------------------------------------


@requires_cuda
def test_cai_and_dlpack_describe_the_same_memory(dplace):
    """Both protocols on one array: same pointer, same geometry. CAI stays a
    borrowed description; DLPack carries ownership."""
    arr, host = _filled((32,), np.float32, dplace)
    cai = arr.__cuda_array_interface__
    t = torch.from_dlpack(arr)
    assert cai["data"][0] == t.data_ptr()
    assert cai["shape"] == tuple(t.shape)
    # a borrowed CAI view (numba) and the owning DLPack tensor stay coherent
    numba_cuda = pytest.importorskip("numba.cuda")
    borrowed = numba_cuda.as_cuda_array(arr)
    t += 1.0
    torch.cuda.synchronize()
    assert np.array_equal(borrowed.copy_to_host(), host + 1.0)


# -- composite (localized) places ---------------------------------------------


@requires_cuda
def test_composite_cute_place_roundtrip():
    """A composite VMM allocation (cute partition over a place grid) exports
    through DLPack like any other array — the owning-tensor lifetime story
    for localized weights."""
    if not hasattr(stf, "cute_partition"):
        pytest.skip("this build predates cute partitions / composite places")
    stf.machine_init()
    grid = stf.exec_place_grid.create([stf.exec_place.device(0)] * 2)
    shape = (8, 64, 16384)  # page-aligned rows (2 MiB at 4 B)
    part = stf.cute_partition.from_spec(shape, (("blocked", 0), None, None), (2,))
    dplace = stf.data_place.composite_cute(grid, part)
    arr = stf.DeviceArray(shape, np.float32, dplace)
    ref = weakref.ref(arr)
    t = torch.from_dlpack(arr)
    del arr
    gc.collect()
    src = torch.randn(shape, device="cuda")
    t.copy_(src)
    torch.cuda.synchronize()
    assert torch.equal(t, src)
    del t
    gc.collect()
    assert ref() is None  # composite VMM freed through the same single point


# -- stream handshake ---------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("stream", [-1, 1, 2, None, "current"])
def test_stream_argument_forms(dplace, stream):
    """All protocol stream encodings are accepted: -1 (no sync), 1/None
    (legacy default), 2 (per-thread default), or a real consumer stream."""
    arr, host = _filled((16,), np.float32, dplace)
    if stream == "current":
        stream = torch.cuda.current_stream().cuda_stream or 1
    cap = arr.__dlpack__(stream=stream)
    assert np.array_equal(torch.from_dlpack(cap).cpu().numpy(), host)


@requires_cuda
def test_stream_ordering_after_producer_stream():
    """Data written on the allocation stream is visible to a consumer stream
    through the event-wait the handshake inserts (no host block)."""
    stf.machine_init()
    dplace = stf.data_place.device(0)
    s_prod = torch.cuda.Stream()
    arr = stf.DeviceArray(1 << 20, np.float32, dplace, stream=s_prod.cuda_stream)
    with torch.cuda.stream(s_prod):
        tmp = torch.full((1 << 20,), 7.0, device="cuda")
    s_cons = torch.cuda.Stream()
    cap = arr.__dlpack__(stream=s_cons.cuda_stream)
    t = torch.from_dlpack(cap)
    with torch.cuda.stream(s_prod):
        t.copy_(tmp)
    cap2 = arr.__dlpack__(stream=s_cons.cuda_stream)
    with torch.cuda.stream(s_cons):
        result = torch.from_dlpack(cap2).sum()
    torch.cuda.synchronize()
    assert float(result) == float(1 << 20) * 7.0


@requires_cuda
def test_invalid_stream_rejected(dplace):
    arr = stf.DeviceArray(4, np.float32, dplace)
    with pytest.raises(ValueError):
        arr.__dlpack__(stream=-7)
    with pytest.raises(TypeError):
        arr.__dlpack__(stream="not-a-stream")


# -- unrepresentable exports --------------------------------------------------


@requires_cuda
def test_structured_dtype_rejected(dplace):
    dtype = np.dtype([("x", np.float32), ("y", np.float32)])
    arr = stf.DeviceArray(4, dtype, dplace)
    assert "descr" in arr.__cuda_array_interface__  # CAI still serves these
    with pytest.raises(BufferError, match="structured"):
        arr.__dlpack__()


@requires_cuda
def test_copy_and_foreign_device_rejected(dplace):
    arr = stf.DeviceArray(4, np.float32, dplace)
    with pytest.raises(BufferError, match="copy"):
        arr.__dlpack__(copy=True)
    with pytest.raises(BufferError, match="device"):
        arr.__dlpack__(dl_device=(1, 0))  # kDLCPU
