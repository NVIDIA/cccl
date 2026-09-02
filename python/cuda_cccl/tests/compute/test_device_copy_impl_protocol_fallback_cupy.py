# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.compute import _device_copy_impl as impl

cp = pytest.importorskip("cupy")


class _NoDLPackArray:
    def __init__(self, array):
        self.array = array
        self.dtype = array.dtype

    @property
    def __cuda_array_interface__(self):
        return self.array.__cuda_array_interface__

    def __dlpack__(self, *args, **kwargs):
        raise BufferError("test array intentionally does not expose DLPack")


def test_DeviceCopyImpl_protocol_fallback_when_dlpack_is_unavailable():
    source = cp.arange(16, dtype=cp.int32)
    destination = cp.empty_like(source)

    impl._copy_into(_NoDLPackArray(source), _NoDLPackArray(destination))

    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(destination, source)
