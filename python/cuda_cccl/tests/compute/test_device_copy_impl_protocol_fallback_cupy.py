# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.compute import _device_copy_impl as impl

cp = pytest.importorskip("cupy")


class _NoDLPackArray:
    def __init__(self, array, *, device_id=None):
        self.array = array
        self.dtype = array.dtype
        self.device_id = cp.cuda.runtime.getDevice() if device_id is None else device_id

    @property
    def __cuda_array_interface__(self):
        return self.array.__cuda_array_interface__

    def __dlpack__(self, *args, **kwargs):
        raise BufferError("test array intentionally does not expose DLPack")

    def __dlpack_device__(self):
        return 2, self.device_id


def test_DeviceCopyImpl_protocol_fallback_when_dlpack_is_unavailable():
    source = cp.arange(16, dtype=cp.int32)
    destination = cp.empty_like(source)

    impl._copy_into(_NoDLPackArray(source), _NoDLPackArray(destination))

    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(destination, source)


def test_DeviceCopyImpl_protocol_fallback_rejects_non_execution_device():
    source = cp.arange(16, dtype=cp.int32)
    destination = cp.empty_like(source)
    reported_device = cp.cuda.runtime.getDevice() + 1

    with pytest.raises(ValueError, match="source is not on the execution device"):
        impl._make_device_copy(
            _NoDLPackArray(source, device_id=reported_device),
            _NoDLPackArray(destination, device_id=reported_device),
        )
