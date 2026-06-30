# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
from _utils.device_array import DeviceArray

from cuda.compute._utils.protocols import (
    get_data_pointer,
    get_dtype,
    get_shape,
    get_size,
    is_contiguous,
)
from cuda.core import Device


@pytest.mark.parametrize("order", ["C", "F"])
def test_empty_exposes_cuda_array_interface(order):
    array = DeviceArray.empty((2, 3), np.float32, order=order)

    expected_strides = np.empty((2, 3), dtype=np.float32, order=order).strides
    interface = array.__cuda_array_interface__

    assert array.nbytes == 24
    assert interface["data"][0] != 0
    assert interface["data"][1] is False
    assert interface["shape"] == (2, 3)
    assert interface["strides"] == (None if order == "C" else expected_strides)
    assert interface["typestr"] == np.dtype(np.float32).str
    assert interface["version"] == 3


def test_cuda_compute_protocol_consumers():
    dtype = np.dtype([("x", np.int32), ("y", np.float32)])
    array = DeviceArray.empty((1, 3), dtype, order="F")
    interface = array.__cuda_array_interface__

    assert get_data_pointer(array) == interface["data"][0]
    assert get_dtype(array) == dtype
    assert get_shape(array) == (1, 3)
    assert get_size(array) == 3
    assert is_contiguous(array)


def test_public_metadata_attributes():
    array = DeviceArray.empty(3, np.int32)

    assert array.nbytes == 12
    assert array.dtype == np.dtype(np.int32)
    assert len(array) == 3
    for attribute in (
        "shape",
        "strides",
        "size",
        "ndim",
        "itemsize",
        "ptr",
        "device_id",
        "stream",
    ):
        assert not hasattr(array, attribute)


def test_scalar_has_no_length():
    array = DeviceArray.empty((), np.int32)

    with pytest.raises(TypeError, match=r"len\(\) of unsized object"):
        len(array)


@pytest.mark.parametrize(
    "host_array",
    [
        np.arange(12, dtype=np.int32).reshape(3, 4),
        np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4)),
        np.asarray(True),
        np.empty((0, 3), dtype=np.int16),
        np.arange(12, dtype=np.int32).reshape(3, 4)[:, ::2],
    ],
)
def test_numpy_round_trip(host_array):
    device_array = DeviceArray.from_numpy(host_array)

    np.testing.assert_array_equal(device_array.copy_to_host(), host_array)
    pointer = device_array.__cuda_array_interface__["data"][0]
    assert (pointer == 0) == (host_array.size == 0)


def test_structured_dtype_round_trip_and_descr():
    dtype = np.dtype([("x", np.int32), ("y", np.float32)])
    host_array = np.asarray([(1, 2.5), (3, 4.5)], dtype=dtype)

    device_array = DeviceArray.from_numpy(host_array)

    np.testing.assert_array_equal(device_array.copy_to_host(), host_array)
    assert device_array.__cuda_array_interface__["descr"] == dtype.descr


def test_aligned_structured_dtype_protocol_round_trip():
    inner_dtype = np.dtype([("x", np.int8), ("y", np.int32)], align=True)
    dtype = np.dtype([("inner", inner_dtype), ("z", np.int16)], align=True)
    device_array = DeviceArray.empty(2, dtype)

    result = get_dtype(device_array)

    assert result == dtype
    assert result.names == dtype.names
    assert result.alignment == dtype.alignment
    assert result.isalignedstruct
    assert result.fields["inner"][0].isalignedstruct


@pytest.mark.parametrize("order", ["C", "F"])
def test_copy_from_host_reuses_allocation_and_validates_metadata(order):
    host_array = np.arange(6, dtype=np.int32).reshape(2, 3)
    device_array = DeviceArray.empty(host_array.shape, host_array.dtype, order=order)
    pointer = device_array.__cuda_array_interface__["data"][0]

    device_array.copy_from_host(host_array)

    assert device_array.__cuda_array_interface__["data"][0] == pointer
    np.testing.assert_array_equal(device_array.copy_to_host(), host_array)

    with pytest.raises(ValueError, match="source shape"):
        device_array.copy_from_host(np.arange(5, dtype=np.int32))
    with pytest.raises(TypeError, match="source dtype"):
        device_array.copy_from_host(host_array.astype(np.int64))


def test_explicit_stream_round_trip():
    device = Device()
    device.set_current()
    stream = device.create_stream()
    host_array = np.arange(8, dtype=np.float32)

    device_array = DeviceArray.from_numpy(host_array, device=device, stream=stream)

    np.testing.assert_array_equal(device_array.copy_to_host(stream=stream), host_array)
