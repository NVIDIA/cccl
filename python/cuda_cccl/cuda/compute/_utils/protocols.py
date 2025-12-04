# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Utilities for extracting information from protocols such as `__cuda_array_interface__` and `__cuda_stream__`.
"""

from typing import List, Optional, Tuple

import numpy as np

from ..typing import DeviceArrayLike, GpuStruct


def get_data_pointer(arr: DeviceArrayLike) -> int:
    # TODO: these are fast paths for CuPy and PyTorch until
    # we have a more general solution.

    # Fast path for PyTorch (arr.data_ptr())
    try:
        return arr.data_ptr()  # type: ignore
    except AttributeError:
        pass

    # Fast path for CuPy (arr.data.ptr)
    try:
        return arr.data.ptr  # type: ignore
    except AttributeError:
        pass

    # Fall back to __cuda_array_interface__
    return arr.__cuda_array_interface__["data"][0]


def get_dtype(arr: DeviceArrayLike | GpuStruct | np.ndarray) -> np.dtype:
    # Try the fast path via .dtype attribute (works for np.ndarray, GpuStruct, and most device arrays)
    try:
        return np.dtype(arr.dtype)  # type: ignore
    except (AttributeError, TypeError):
        pass

    # Fall back to __cuda_array_interface__ for DeviceArrayLike
    cai = arr.__cuda_array_interface__  # type: ignore
    typestr = cai["typestr"]

    if typestr.startswith("|V"):
        # it's a structured dtype, use the descr field:
        return np.dtype(cai["descr"])
    else:
        # a simple dtype, use the typestr field:
        return np.dtype(typestr)


def get_shape(arr: DeviceArrayLike) -> Tuple[int]:
    try:
        # TODO: this is a fast path for CuPy until
        # we have a more general solution.
        return arr.shape  # type: ignore
    except AttributeError:
        return arr.__cuda_array_interface__["shape"]


def is_contiguous(arr: DeviceArrayLike) -> bool:
    cai = arr.__cuda_array_interface__

    strides = cai["strides"]

    if strides is None:
        return True

    shape = cai["shape"]

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


def compute_c_contiguous_strides_in_bytes(
    shape: Tuple[int], itemsize: int
) -> Tuple[int, ...]:
    """Return C-contiguous strides in bytes for a given shape and itemsize (compatible with NumPy .strides)."""

    strides: List[int] = []
    acc = itemsize

    for dim in reversed(shape):
        strides.insert(0, acc)
        acc *= dim

    return tuple(strides)


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
