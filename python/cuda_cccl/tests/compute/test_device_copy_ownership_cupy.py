# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc
import importlib
import threading
import weakref

import pytest


def _import_or_skip(module_name):
    try:
        return importlib.import_module(module_name)
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"{module_name} is unavailable: {exc}")


@pytest.fixture(scope="module")
def cp():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CuPy did not find a CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy CUDA runtime is unavailable: {exc}")
    return cupy


@pytest.fixture(scope="module")
def device_copy_impl():
    return _import_or_skip("cuda.compute._device_copy_impl")


class _StreamProtocol:
    def __init__(self, stream):
        self._stream = stream

    def __cuda_stream__(self):
        return 0, self._stream.ptr


def test_DeviceCopy_keeps_dlpack_owners_alive_until_stream_completion(
    cp, device_copy_impl
):
    source = cp.arange(256, dtype=cp.int32)
    destination = cp.empty_like(source)
    device_copy = device_copy_impl._make_device_copy(source, destination)
    stream = cp.cuda.Stream(non_blocking=True)
    stream_protocol = _StreamProtocol(stream)
    callback_started = threading.Event()
    release_stream = threading.Event()

    def block_stream(_argument):
        callback_started.set()
        release_stream.wait()

    try:
        stream.launch_host_func(block_stream, None)
        assert callback_started.wait(timeout=10)

        device_copy(source, destination, stream=stream_protocol)
        source_ref = weakref.ref(source)
        destination_ref = weakref.ref(destination)
        del source
        del destination
        gc.collect()

        assert source_ref() is not None
        assert destination_ref() is not None

        release_stream.set()
        stream.synchronize()
        device_copy_impl._drain_device_copy_owner_releases()
        gc.collect()

        assert source_ref() is None
        assert destination_ref() is None
    finally:
        release_stream.set()
        stream.synchronize()
        device_copy_impl._drain_device_copy_owner_releases()
        device_copy.close()
