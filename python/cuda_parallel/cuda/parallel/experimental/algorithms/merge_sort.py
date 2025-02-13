# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
from typing import Callable

import numba
from numba import cuda
from numba.cuda.cudadrv import enums

from .. import _cccl as cccl
from .._bindings import get_bindings, get_paths
from .._utils import protocols
from ..iterators._iterators import IteratorBase
from ..typing import DeviceArrayLike


def _dtype_validation(dt1, dt2):
    if dt1 != dt2:
        raise TypeError(f"dtype mismatch: __init__={dt1}, __call__={dt2}")


class _MergeSort:
    def __init__(
        self,
        d_in_keys: DeviceArrayLike | IteratorBase,
        d_in_items: DeviceArrayLike | IteratorBase | None,
        d_out_keys: DeviceArrayLike,
        d_out_items: DeviceArrayLike | None,
        op: Callable,
    ):
        assert (d_in_items is None) == (d_out_items is None)

        # Referenced from __del__:
        self.build_result = None

        d_in_keys_cccl = cccl.to_cccl_iter(d_in_keys)
        d_in_items_cccl = cccl.to_cccl_iter(d_in_items)
        d_out_keys_cccl = (
            d_in_keys_cccl if d_in_keys is d_out_keys else cccl.to_cccl_iter(d_out_keys)
        )
        d_out_items_cccl = (
            d_in_items_cccl
            if d_in_items is d_out_items
            else cccl.to_cccl_iter(d_out_items)
        )

        self._ctor_d_in_keys_cccl_type = cccl.type_enum_as_name(
            d_in_keys_cccl.value_type.type.value
        )
        self._ctor_d_in_items_cccl_type = cccl.type_enum_as_name(
            d_in_items_cccl.value_type.type.value
        )
        self._ctor_d_out_keys_cccl_type = cccl.type_enum_as_name(
            d_out_keys_cccl.value_type.type.value
        )
        self._ctor_d_out_items_cccl_type = cccl.type_enum_as_name(
            d_out_items_cccl.value_type.type.value
        )

        cc_major, cc_minor = cuda.get_current_device().compute_capability
        cub_path, thrust_path, libcudacxx_path, cuda_include_path = get_paths()
        bindings = get_bindings()

        if isinstance(d_in_keys, IteratorBase):
            value_type = d_in_keys.value_type
        else:
            value_type = numba.from_dtype(protocols.get_dtype(d_in_keys))

        sig = (value_type, value_type)
        self.op_wrapper = cccl.to_cccl_op(op, sig)

        self.build_result = cccl.DeviceMergeSortBuildResult()
        error = bindings.cccl_device_merge_sort_build(
            ctypes.byref(self.build_result),
            d_in_keys_cccl,
            d_in_items_cccl,
            d_out_keys_cccl,
            d_out_items_cccl,
            self.op_wrapper,
            cc_major,
            cc_minor,
            ctypes.c_char_p(cub_path),
            ctypes.c_char_p(thrust_path),
            ctypes.c_char_p(libcudacxx_path),
            ctypes.c_char_p(cuda_include_path),
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error building merge_sort")

    # TODO should I pass op here as well
    def __call__(
        self,
        temp_storage,
        d_in_keys: DeviceArrayLike | IteratorBase,
        d_in_items: DeviceArrayLike | IteratorBase | None,
        d_out_keys: DeviceArrayLike,
        d_out_items: DeviceArrayLike | None,
        num_items: int,
        stream=None,
    ):
        assert (d_in_items is None) == (d_out_items is None)

        d_in_keys_cccl = cccl.to_cccl_iter(d_in_keys)
        d_in_items_cccl = cccl.to_cccl_iter(d_in_items)
        d_out_keys_cccl = (
            d_in_keys_cccl if d_in_keys is d_out_keys else cccl.to_cccl_iter(d_out_keys)
        )
        d_out_items_cccl = (
            d_in_items_cccl
            if d_in_items is d_out_items
            else cccl.to_cccl_iter(d_out_items)
        )

        _dtype_validation(
            self._ctor_d_in_keys_cccl_type,
            cccl.type_enum_as_name(d_in_keys_cccl.value_type.type.value),
        )
        _dtype_validation(
            self._ctor_d_in_items_cccl_type,
            cccl.type_enum_as_name(d_in_items_cccl.value_type.type.value),
        )
        _dtype_validation(
            self._ctor_d_out_keys_cccl_type,
            cccl.type_enum_as_name(d_out_keys_cccl.value_type.type.value),
        )
        _dtype_validation(
            self._ctor_d_out_items_cccl_type,
            cccl.type_enum_as_name(d_out_items_cccl.value_type.type.value),
        )

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

        error = bindings.cccl_device_merge_sort(
            self.build_result,
            ctypes.c_void_p(d_temp_storage),
            ctypes.byref(temp_storage_bytes),
            d_in_keys_cccl,
            d_in_items_cccl,
            d_out_keys_cccl,
            d_out_items_cccl,
            ctypes.c_ulonglong(num_items),
            self.op_wrapper,
            ctypes.c_void_p(stream_handle),
        )

        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error in merge sort")

        return temp_storage_bytes.value

    def __del__(self):
        if self.build_result is None:
            return
        bindings = get_bindings()
        bindings.cccl_device_merge_sort_cleanup(ctypes.byref(self.build_result))


def merge_sort(
    d_in_keys: DeviceArrayLike | IteratorBase,
    d_in_items: DeviceArrayLike | IteratorBase | None,
    d_out_keys: DeviceArrayLike,
    d_out_items: DeviceArrayLike | None,
    op: Callable,
):
    """Implements a device-wide merge sort using ``d_in_keys`` and the comparison operator ``op``.

    Example:
        Below, ``merge_sort`` is used to sort a sequence of keys inplace. It also rearranges the items according to the keys' order.

        .. literalinclude:: ../../python/cuda_parallel/tests/test_merge_sort_api.py
          :language: python
          :dedent:
          :start-after: example-begin merge-sort
          :end-before: example-end merge-sort

    Args:
        d_in_keys: Device array or iterator containing the input keys to be sorted
        d_in_items: Optional device array or iterator that contains each key's corresponding item
        d_in_keys: Device array to store the sorted keys
        d_in_items: Device array to store the sorted items
        op: Callable representing the comparison operator

    Returns:
        A callable object that can be used to perform the merge sort
    """
    return _MergeSort(d_in_keys, d_in_items, d_out_keys, d_out_items, op)
