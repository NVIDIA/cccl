# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Utilities for extracting information from protocols such as `__cuda_array_interface__` and `__cuda_stream__`.
"""

from typing import Optional, Tuple

import numpy as np

from ..typing import DeviceArrayLike


def get_dtype(arr: DeviceArrayLike) -> np.dtype:
    typestr = arr.__cuda_array_interface__["typestr"]

    if typestr.startswith("|V"):
        # it's a structured dtype, use the descr field:
        return np.dtype(arr.__cuda_array_interface__["descr"])
    else:
        # a simple dtype, use the typestr field:
        return np.dtype(typestr)


def get_strides(arr: DeviceArrayLike) -> Optional[Tuple]:
    return arr.__cuda_array_interface__["strides"]


def get_shape(arr: DeviceArrayLike) -> Tuple:
    return arr.__cuda_array_interface__["shape"]


def is_contiguous(arr: DeviceArrayLike) -> bool:
    shape, strides = get_shape(arr), get_strides(arr)

    if strides is None:
        return True

    if any(dim == 0 for dim in shape):
        # array has no elements
        return True

    if all(dim == 1 for dim in shape):
        # there is a single element:
        return True

    itemsize = get_dtype(arr).itemsize

    if strides[-1] == itemsize:
        # assume C-contiguity
        expected_stride = itemsize
        for dim, stride in zip(reversed(shape), reversed(strides)):
            if stride != expected_stride:
                return False
            expected_stride *= dim
        return True
    elif strides[0] == itemsize:
        # assume F-contiguity
        expected_stride = itemsize
        for dim, stride in zip(shape, strides):
            if stride != expected_stride:
                return False
            expected_stride *= dim
        return True
    else:
        # not contiguous
        return False


def validate_and_get_stream(stream) -> Optional[int]:
    # null stream is allowed
    if stream is None:
        return None

    try:
        stream_property = stream.__cuda_stream__()
    except AttributeError as e:
        raise TypeError(
            f"stream argument {stream} does not implement the '__cuda_stream__' protocol"
        ) from e

    try:
        version, handle, *_ = stream_property
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"could not obtain __cuda_stream__ protocol version and handle from {stream_property}"
        ) from e

    if version == 0:
        if not isinstance(handle, int):
            raise TypeError(f"invalid stream handle {handle}")
        return handle

    raise TypeError(f"unsupported __cuda_stream__ version {version}")
