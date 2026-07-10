# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for exec_place.from_context (externally-owned CUDA contexts) and the
cuda.core-backed green_places() helper."""

import weakref

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402


def _require_cuda_core_device():
    try:
        from cuda.core import Device
    except ImportError:
        pytest.skip("cuda-core is not available")
    try:
        dev = Device(0)
        dev.set_current()
    except Exception as exc:
        pytest.skip(f"no usable CUDA device: {exc}")
    return dev


def _require_green_context(dev, sm_count=8):
    try:
        from cuda.core._context import ContextOptions
        from cuda.core._device_resources import SMResourceOptions
    except ImportError:
        pytest.skip("cuda-core >= 1.0 with green-context support is required")
    try:
        groups, _remainder = dev.resources.sm.split(SMResourceOptions(count=sm_count))
    except Exception as exc:
        pytest.skip(f"green context support unavailable: {exc}")
    if not groups:
        pytest.skip("device SM resource could not be split")
    return dev.create_context(ContextOptions(resources=[groups[0]]))


def test_from_context_primary_context():
    """A place from the device's primary context behaves like a device place."""
    stf.machine_init()
    _require_cuda_core_device()
    from cuda.bindings import driver as cu

    err, dev = cu.cuDeviceGet(0)
    assert err == cu.CUresult.CUDA_SUCCESS
    err, ctx = cu.cuDevicePrimaryCtxRetain(dev)
    assert err == cu.CUresult.CUDA_SUCCESS
    try:
        place = stf.exec_place.from_context(int(ctx))
        assert place.kind == "device"
        assert place.affine_data_place.device_id == 0

        resources = stf.exec_place_resources()
        with place:
            stream = place.pick_stream(resources)
            assert isinstance(stream, stf.CudaStream)
            assert stream != 0
    finally:
        cu.cuDevicePrimaryCtxRelease(dev)


def test_from_context_cuda_core_green_context():
    """A place built from a cuda.core green context: devid derivation + streams."""
    stf.machine_init()
    dev = _require_cuda_core_device()
    ctx = _require_green_context(dev)
    assert ctx.is_green

    # dev_id intentionally omitted: derived from the context
    place = stf.exec_place.from_context(ctx)
    assert place.kind == "device"
    assert place.affine_data_place.device_id == 0

    resources = stf.exec_place_resources()
    with place:
        stream = place.pick_stream(resources)
        assert isinstance(stream, stf.CudaStream)
        assert stream != 0


def test_from_context_keeps_backing_object_alive():
    """The place must hold a reference to the cuda.core Context."""
    stf.machine_init()
    dev = _require_cuda_core_device()
    ctx = _require_green_context(dev)
    ref = weakref.ref(ctx)

    place = stf.exec_place.from_context(ctx)
    del ctx
    assert ref() is not None, "place dropped the backing Context"

    del place
    # Not asserting collection here (GC timing), only that deletion is safe.


def test_from_context_rejects_bad_input():
    stf.machine_init()
    with pytest.raises(ValueError):
        stf.exec_place.from_context(0)
    with pytest.raises(TypeError):
        stf.exec_place.from_context("not a context")


def test_from_context_task_roundtrip():
    """Run an actual STF task on a green-context place and verify the result."""
    stf.machine_init()
    dev = _require_cuda_core_device()
    ctx = _require_green_context(dev)
    place = stf.exec_place.from_context(ctx)

    X = np.arange(64, dtype=np.float64)
    expected = X * 2.0

    cp = pytest.importorskip("cupy")

    sctx = stf.context()
    lX = sctx.logical_data(X, name="X")
    with sctx.task(place, lX.rw()) as t:
        with cp.cuda.ExternalStream(int(t.stream_ptr())):
            dX = cp.asarray(t.get_arg_cai(0))
            dX *= 2.0
    sctx.finalize()

    np.testing.assert_allclose(X, expected)


def test_green_places_helper():
    """green_places() returns working places backed by green contexts."""
    stf.machine_init()
    _require_cuda_core_device()

    try:
        places = stf.green_places(sms_per_place=8, n_places=2)
    except RuntimeError as exc:
        pytest.skip(f"green_places unavailable: {exc}")

    assert len(places) == 2
    resources = stf.exec_place_resources()
    streams = []
    for place in places:
        assert place.kind == "device"
        with place:
            streams.append(place.pick_stream(resources))
    assert all(s != 0 for s in streams)
