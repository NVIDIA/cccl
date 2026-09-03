# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.compute import _device_copy_impl as impl

cp = pytest.importorskip("cupy")


def test_DeviceCopy_get_source_returns_generated_source():
    source = cp.arange(16, dtype=cp.int32)
    destination = cp.empty_like(source)

    device_copy = impl._make_device_copy(source, destination)

    generated_source = device_copy._get_source()

    assert isinstance(generated_source, str)
    assert (
        'extern "C" _CCCL_VISIBILITY_EXPORT int cccl_jit_device_copy'
        in generated_source
    )
    assert "cub::DeviceCopy::Copy" in generated_source
