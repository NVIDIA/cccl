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


class _Op:
    def __init__(self, h_init: np.ndarray | GpuStruct, op: Callable):
        if isinstance(h_init, np.ndarray):
            value_type = numba.from_dtype(h_init.dtype)
        else:
            value_type = numba.typeof(h_init)
        self.ltoir, _ = cuda.compile(op, sig=(value_type, value_type), output="ltoir")
        self.name = op.__name__.encode("utf-8")

    def handle(self) -> cccl.Op:
        return cccl.Op(
            cccl.OpKind.STATELESS,
            self.name,
            ctypes.c_char_p(self.ltoir),
            len(self.ltoir),
            1,
            1,
            None,
        )


def _dtype_validation(dt1, dt2):
    if dt1 != dt2:
        raise TypeError(f"dtype mismatch: __init__={dt1}, __call__={dt2}")


class _Reduce:
    # TODO: constructor shouldn't require concrete `d_in`, `d_out`:
    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike,
        op: Callable,
        h_init: np.ndarray | GpuStruct,
    ):
        # Referenced from __del__:
        self.build_result = None

        d_in_cccl = cccl.to_cccl_iter(d_in)
        self._ctor_d_in_cccl_type_enum_name = cccl.type_enum_as_name(
            d_in_cccl.value_type.type.value
        )
        self._ctor_d_out_dtype = protocols.get_dtype(d_out)
        self._ctor_init_dtype = h_init.dtype
        cc_major, cc_minor = cuda.get_current_device().compute_capability
        cub_path, thrust_path, libcudacxx_path, cuda_include_path = get_paths()
        bindings = get_bindings()
        self.op_wrapper = _Op(h_init, op)
        d_out_cccl = cccl.to_cccl_iter(d_out)
        self.build_result = cccl.DeviceReduceBuildResult()

        error = bindings.cccl_device_reduce_build(
            ctypes.byref(self.build_result),
            d_in_cccl,
            d_out_cccl,
            self.op_wrapper.handle(),
            cccl.to_cccl_value(h_init),
            cc_major,
            cc_minor,
            ctypes.c_char_p(cub_path),
            ctypes.c_char_p(thrust_path),
            ctypes.c_char_p(libcudacxx_path),
            ctypes.c_char_p(cuda_include_path),
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error building reduce")

    def __call__(
        self,
        temp_storage,
        d_in,
        d_out,
        num_items: int,
        h_init: np.ndarray | GpuStruct,
        stream=None,
    ):
        d_in_cccl = cccl.to_cccl_iter(d_in)
        if d_in_cccl.type.value == cccl.IteratorKind.ITERATOR:
            assert num_items is not None
        else:
            assert d_in_cccl.type.value == cccl.IteratorKind.POINTER
            if num_items is None:
                num_items = d_in.size
            else:
                assert num_items == d_in.size
        _dtype_validation(
            self._ctor_d_in_cccl_type_enum_name,
            cccl.type_enum_as_name(d_in_cccl.value_type.type.value),
        )
        _dtype_validation(self._ctor_d_out_dtype, protocols.get_dtype(d_out))
        _dtype_validation(self._ctor_init_dtype, h_init.dtype)
        stream_handle = protocols.validate_and_get_stream(stream)
        bindings = get_bindings()
        if temp_storage is None:
            temp_storage_bytes = ctypes.c_size_t()
            d_temp_storage = None
        else:
            temp_storage_bytes = ctypes.c_size_t(temp_storage.nbytes)
            # Note: this is slightly slower, but supports all ndarray-like objects as long as they support CAI
            # TODO: switch to use gpumemoryview once it's ready
            d_temp_storage = temp_storage.__cuda_array_interface__["data"][0]
        d_out_cccl = cccl.to_cccl_iter(d_out)
        error = bindings.cccl_device_reduce(
            self.build_result,
            d_temp_storage,
            ctypes.byref(temp_storage_bytes),
            d_in_cccl,
            d_out_cccl,
            ctypes.c_ulonglong(num_items),
            self.op_wrapper.handle(),
            cccl.to_cccl_value(h_init),
            stream_handle,
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error reducing")

        return temp_storage_bytes.value

    def __del__(self):
        if self.build_result is None:
            return
        bindings = get_bindings()
        bindings.cccl_device_reduce_cleanup(ctypes.byref(self.build_result))


def make_cache_key(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    op: Callable,
    h_init: np.ndarray,
):
    d_in_key = (
        d_in.kind if isinstance(d_in, IteratorBase) else protocols.get_dtype(d_in)
    )
    d_out_key = protocols.get_dtype(d_out)
    op_key = CachableFunction(op)
    h_init_key = h_init.dtype
    return (d_in_key, d_out_key, op_key, h_init_key)


# TODO Figure out `sum` without operator and initial value
# TODO Accept stream
@cache_with_key(make_cache_key)
def reduce_into(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    op: Callable,
    h_init: np.ndarray,
):
    """Computes a device-wide reduction using the specified binary ``op`` functor and initial value ``init``.

    Example:
        The code snippet below demonstrates the usage of the ``reduce_into`` API:

        .. literalinclude:: ../../python/cuda_parallel/tests/test_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce-min
            :end-before: example-end reduce-min

    Args:
        d_in: CUDA device array storing the input sequence of data items
        d_out: CUDA device array storing the output aggregate
        op: Binary reduction
        init: Numpy array storing initial value of the reduction

    Returns:
        A callable object that can be used to perform the reduction
    """
    return _Reduce(d_in, d_out, op, h_init)
