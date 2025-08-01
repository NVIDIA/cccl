# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations  # TODO: required for Python 3.7 docs env

from typing import Callable

import numba
import numpy as np

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import CachableFunction, cache_with_key
from .._cccl_interop import call_build, set_cccl_iterator_state, to_cccl_value_state
from .._utils import protocols
from .._utils.protocols import get_data_pointer, validate_and_get_stream
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..iterators._iterators import IteratorBase
from ..typing import DeviceArrayLike, GpuStruct


class _Scan:
    __slots__ = [
        "build_result",
        "d_in_cccl",
        "d_out_cccl",
        "h_init_cccl",
        "op_wrapper",
        "device_scan_fn",
    ]

    # TODO: constructor shouldn't require concrete `d_in`, `d_out`:
    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike | IteratorBase,
        op: Callable,
        h_init: np.ndarray | GpuStruct,
        force_inclusive: bool,
    ):
        self.d_in_cccl = cccl.to_cccl_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_iter(d_out)
        self.h_init_cccl = cccl.to_cccl_value(h_init)
        if isinstance(h_init, np.ndarray):
            value_type = numba.from_dtype(h_init.dtype)
        else:
            value_type = numba.typeof(h_init)
        self.op_wrapper = cccl.to_cccl_op(op, value_type(value_type, value_type))
        self.build_result = call_build(
            _bindings.DeviceScanBuildResult,
            self.d_in_cccl,
            self.d_out_cccl,
            self.op_wrapper,
            self.h_init_cccl,
            force_inclusive,
        )

        self.device_scan_fn = (
            self.build_result.compute_inclusive
            if force_inclusive
            else self.build_result.compute_exclusive
        )

    def __call__(
        self,
        temp_storage,
        d_in,
        d_out,
        num_items: int,
        h_init: np.ndarray | GpuStruct,
        stream=None,
    ):
        set_cccl_iterator_state(self.d_in_cccl, d_in)
        set_cccl_iterator_state(self.d_out_cccl, d_out)

        self.h_init_cccl.state = to_cccl_value_state(h_init)

        stream_handle = validate_and_get_stream(stream)

        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            d_temp_storage = get_data_pointer(temp_storage)

        temp_storage_bytes = self.device_scan_fn(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_cccl,
            self.d_out_cccl,
            num_items,
            self.op_wrapper,
            self.h_init_cccl,
            stream_handle,
        )
        return temp_storage_bytes


def make_cache_key(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable,
    h_init: np.ndarray,
):
    d_in_key = (
        d_in.kind if isinstance(d_in, IteratorBase) else protocols.get_dtype(d_in)
    )
    d_out_key = (
        d_out.kind if isinstance(d_out, IteratorBase) else protocols.get_dtype(d_out)
    )
    op_key = CachableFunction(op)
    h_init_key = h_init.dtype
    return (d_in_key, d_out_key, op_key, h_init_key)


# TODO Figure out `sum` without operator and initial value
# TODO Accept stream
@cache_with_key(make_cache_key)
def make_exclusive_scan(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable,
    h_init: np.ndarray,
):
    """Computes a device-wide scan using the specified binary ``op`` and initial value ``init``.

    Example:
        Below, ``scan`` is used to compute an exclusive scan of a sequence of integers.

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_scan_api.py
          :language: python
          :dedent:
          :start-after: example-begin exclusive-scan-max
          :end-before: example-end exclusive-scan-max

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array that will store the result of the scan
        op: Callable representing the binary operator to apply
        init: Numpy array storing initial value of the scan

    Returns:
        A callable object that can be used to perform the scan
    """
    return _Scan(d_in, d_out, op, h_init, False)


def exclusive_scan(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable,
    h_init: np.ndarray | GpuStruct,
    num_items: int,
    stream=None,
):
    scanner = make_exclusive_scan(d_in, d_out, op, h_init)
    tmp_storage_bytes = scanner(None, d_in, d_out, num_items, h_init, stream)
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    scanner(tmp_storage, d_in, d_out, num_items, h_init, stream)


# TODO Figure out `sum` without operator and initial value
# TODO Accept stream
@cache_with_key(make_cache_key)
def make_inclusive_scan(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable,
    h_init: np.ndarray,
):
    """Computes a device-wide scan using the specified binary ``op`` and initial value ``init``.

    Example:
        Below, ``scan`` is used to compute an inclusive scan of a sequence of integers.

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_scan_api.py
          :language: python
          :dedent:
          :start-after: example-begin inclusive-scan-add
          :end-before: example-end inclusive-scan-add

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array that will store the result of the scan
        op: Callable representing the binary operator to apply
        init: Numpy array storing initial value of the scan

    Returns:
        A callable object that can be used to perform the scan
    """
    return _Scan(d_in, d_out, op, h_init, True)


def inclusive_scan(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable,
    h_init: np.ndarray | GpuStruct,
    num_items: int,
    stream=None,
):
    scanner = make_inclusive_scan(d_in, d_out, op, h_init)
    tmp_storage_bytes = scanner(None, d_in, d_out, num_items, h_init, stream)
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    scanner(tmp_storage, d_in, d_out, num_items, h_init, stream)
