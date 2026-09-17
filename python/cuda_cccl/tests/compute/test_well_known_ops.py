# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute import OpKind, gpu_struct, make_binary_transform

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

pytestmark = pytest.mark.no_numba


@pytest.mark.parametrize("structured", [True, False], ids=["struct", "complex"])
def test_identity_with_storage(structured):
    Point = gpu_struct({"x": np.int32, "y": np.int32})
    if structured:
        h_in = np.array([(1, 2), (-3, 4), (5, -6), (7, 8)], dtype=Point.dtype)
    else:
        h_in = np.array([1 + 2j, -3 + 4j, 5 - 6j, 7 + 8j], dtype=np.complex64)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)

    if USING_V2:
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=OpKind.IDENTITY, num_items=h_in.size
        )
        np.testing.assert_array_equal(d_out.copy_to_host(), h_in)
    else:
        with pytest.raises(
            TypeError,
            match=r"OpKind\.IDENTITY is not supported for struct or other opaque types.*"
            r"Provide a custom operator instead\.",
        ):
            cuda.compute.unary_transform(
                d_in=d_in, d_out=d_out, op=OpKind.IDENTITY, num_items=h_in.size
            )


@pytest.mark.skipif(USING_V2, reason="storage type rejection is specific to V1")
@pytest.mark.parametrize("storage_position", [0, 1, 2], ids=["in1", "in2", "out"])
def test_binary_transform_with_storage(storage_position):
    Point = gpu_struct({"x": np.int32, "y": np.int32})
    dtypes = [np.int32, np.int32, np.int32]
    dtypes[storage_position] = Point.dtype
    d_in1, d_in2, d_out = [DeviceArray.empty(4, dtype) for dtype in dtypes]

    with pytest.raises(
        TypeError,
        match=r"OpKind\.PLUS is not supported for struct or other opaque types.*"
        r"Provide a custom operator instead\.",
    ):
        make_binary_transform(d_in1=d_in1, d_in2=d_in2, d_out=d_out, op=OpKind.PLUS)


def test_sort_with_struct_values():
    # Only the keys participate in the built-in comparison.
    Point = gpu_struct({"x": np.int32, "y": np.int32})
    h_keys = np.array([3, 1, 4, 2], dtype=np.int32)
    h_values = np.array([(30, 31), (10, 11), (40, 41), (20, 21)], dtype=Point.dtype)
    d_in_keys = DeviceArray.from_numpy(h_keys)
    d_in_values = DeviceArray.from_numpy(h_values)
    d_out_keys = DeviceArray.empty(h_keys.shape, h_keys.dtype)
    d_out_values = DeviceArray.empty(h_values.shape, h_values.dtype)

    cuda.compute.merge_sort(
        d_in_keys=d_in_keys,
        d_in_values=d_in_values,
        d_out_keys=d_out_keys,
        d_out_values=d_out_values,
        num_items=h_keys.size,
        op=OpKind.LESS,
    )

    order = np.argsort(h_keys)
    np.testing.assert_array_equal(d_out_keys.copy_to_host(), h_keys[order])
    np.testing.assert_array_equal(d_out_values.copy_to_host(), h_values[order])
