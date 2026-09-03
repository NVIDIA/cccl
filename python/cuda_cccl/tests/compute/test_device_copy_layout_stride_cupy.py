# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.compute import _device_copy_impl as impl

cp = pytest.importorskip("cupy")


def test_DeviceCopy_uses_layout_stride_for_positive_runtime_strides():
    source_base = cp.arange(24, dtype=cp.int32).reshape(4, 6)
    source = source_base[:, ::2]
    destination_base = cp.empty((3, 4), dtype=cp.int32)
    destination = destination_base.T

    device_copy = impl._make_device_copy(source, destination)

    generated_source = device_copy._get_source()
    device_copy(source, destination)

    assert "using input_type = ::cuda::std::mdspan" in generated_source
    assert "using output_type = ::cuda::std::mdspan" in generated_source
    assert "::cuda::std::layout_stride" in generated_source
    assert "source_view_mapping_type{extents, source_view_strides}" in generated_source
    assert (
        "destination_view_mapping_type{extents, destination_view_strides}"
        in generated_source
    )
    cp.testing.assert_array_equal(destination, source)
