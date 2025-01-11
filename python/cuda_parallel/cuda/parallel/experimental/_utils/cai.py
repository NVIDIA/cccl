# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Utilities for extracting information from `__cuda_array_interface__`.
"""

from typing import Optional, Tuple

import numpy as np

from ..typing import DeviceArrayLike


def get_dtype(arr: DeviceArrayLike) -> np.dtype:
    return np.dtype(arr.__cuda_array_interface__["typestr"])


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
