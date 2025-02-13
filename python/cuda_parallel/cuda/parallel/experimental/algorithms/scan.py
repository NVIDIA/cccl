# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations  # TODO: required for Python 3.7 docs env

import ctypes
from typing import Callable

import numba
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import enums

from .. import _cccl as cccl
from .._bindings import get_bindings, get_paths
from .._caching import CachableFunction, cache_with_key
from .._utils import protocols
from ..iterators._iterators import IteratorBase
from ..typing import DeviceArrayLike, GpuStruct


class _Scan:
    # TODO: constructor shouldn't require concrete `d_in`, `d_out`:
    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike | IteratorBase,
        op: Callable,
        h_init: np.ndarray | GpuStruct,
    ):
        # Referenced from __del__:
        self.build_result = None

        self.d_in_cccl = cccl.to_cccl_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_iter(d_out)
        self.h_init_cccl = cccl.to_cccl_value(h_init)
        cc_major, cc_minor = cuda.get_current_device().compute_capability
        cub_path, thrust_path, libcudacxx_path, cuda_include_path = get_paths()
        if isinstance(h_init, np.ndarray):
            value_type = numba.from_dtype(h_init.dtype)
        else:
            value_type = numba.typeof(h_init)
        sig = (value_type, value_type)
        self.op_wrapper = cccl.to_cccl_op(op, sig)
        self.build_result = cccl.DeviceScanBuildResult()
        self.bindings = get_bindings()
        error = self.bindings.cccl_device_scan_build(
            ctypes.byref(self.build_result),
            self.d_in_cccl,
            self.d_out_cccl,
            self.op_wrapper,
            cccl.to_cccl_value(h_init),
            cc_major,
            cc_minor,
            ctypes.c_char_p(cub_path),
            ctypes.c_char_p(thrust_path),
            ctypes.c_char_p(libcudacxx_path),
            ctypes.c_char_p(cuda_include_path),
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error building scan")

    def __call__(
        self,
        temp_storage,
        d_in,
        d_out,
        num_items: int,
        h_init: np.ndarray | GpuStruct,
        stream=None,
    ):
        if self.d_in_cccl.type.value == cccl.IteratorKind.POINTER:
            self.d_in_cccl.state = protocols.get_data_pointer(d_in)
        else:
            self.d_in_cccl.state = d_in.state

        if self.d_out_cccl.type.value == cccl.IteratorKind.POINTER:
            self.d_out_cccl.state = protocols.get_data_pointer(d_out)
        else:
            self.d_out_cccl.state = d_out.state

        self.h_init_cccl.state = h_init.__array_interface__["data"][0]

        stream_handle = protocols.validate_and_get_stream(stream)

        if temp_storage is None:
            temp_storage_bytes = ctypes.c_size_t()
            d_temp_storage = None
        else:
            temp_storage_bytes = ctypes.c_size_t(temp_storage.nbytes)
            d_temp_storage = protocols.get_data_pointer(temp_storage)

        error = self.bindings.cccl_device_scan(
            self.build_result,
            ctypes.c_void_p(d_temp_storage),
            ctypes.byref(temp_storage_bytes),
            self.d_in_cccl,
            self.d_out_cccl,
            ctypes.c_ulonglong(num_items),
            self.op_wrapper,
            self.h_init_cccl,
            ctypes.c_void_p(stream_handle),
        )

        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error reducing")

        return temp_storage_bytes.value

    def __del__(self):
        if self.build_result is None:
            return
        bindings = get_bindings()
        bindings.cccl_device_scan_cleanup(ctypes.byref(self.build_result))


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
def scan(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable,
    h_init: np.ndarray,
):
    """Computes a device-wide scan using the specified binary ``op`` and initial value ``init``.

    Example:
        Below, ``scan`` is used to compute an exclusive scan of a sequence of integers.

        .. literalinclude:: ../../python/cuda_parallel/tests/test_scan_api.py
          :language: python
          :dedent:
          :start-after: example-begin scan-max
          :end-before: example-end scan-max

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array that will store the result of the scan
        op: Callable representing the binary operator to apply
        init: Numpy array storing initial value of the scan

    Returns:
        A callable object that can be used to perform the scan
    """
    return _Scan(d_in, d_out, op, h_init)
