# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
from typing import Callable

import numba
from numba.cuda.cudadrv import enums

from .. import _cccl as cccl
from .._bindings import call_build, get_bindings
from .._caching import CachableFunction, cache_with_key
from .._utils import protocols
from ..iterators_legacy._iterators import IteratorBase
from ..typing import DeviceArrayLike


def make_cache_key(
    d_in_keys: DeviceArrayLike | IteratorBase,
    d_in_items: DeviceArrayLike | IteratorBase,
    d_out_keys: DeviceArrayLike | IteratorBase,
    d_out_items: DeviceArrayLike | IteratorBase,
    d_out_num_selected: DeviceArrayLike,
    op: Callable,
):
    d_in_keys_key = (
        d_in_keys.kind
        if isinstance(d_in_keys, IteratorBase)
        else protocols.get_dtype(d_in_keys)
    )
    d_in_items_key = (
        d_in_items.kind
        if isinstance(d_in_items, IteratorBase)
        else protocols.get_dtype(d_in_items)
    )
    d_out_keys_key = (
        d_out_keys.kind
        if isinstance(d_out_keys, IteratorBase)
        else protocols.get_dtype(d_out_keys)
    )
    d_out_items_key = (
        d_out_items.kind
        if isinstance(d_out_items, IteratorBase)
        else protocols.get_dtype(d_out_items)
    )
    d_out_num_selected_key = protocols.get_dtype(d_out_num_selected)
    op_key = CachableFunction(op)

    return (
        d_in_keys_key,
        d_in_items_key,
        d_out_keys_key,
        d_out_items_key,
        d_out_num_selected_key,
        op_key,
    )


class _UniqueByKey:
    def __init__(
        self,
        d_in_keys: DeviceArrayLike | IteratorBase,
        d_in_items: DeviceArrayLike | IteratorBase,
        d_out_keys: DeviceArrayLike | IteratorBase,
        d_out_items: DeviceArrayLike | IteratorBase,
        d_out_num_selected: DeviceArrayLike,
        op: Callable,
    ):
        # Referenced from __del__:
        self.build_result = None

        self.d_in_keys_cccl = cccl.to_cccl_iter(d_in_keys)
        self.d_in_items_cccl = cccl.to_cccl_iter(d_in_items)
        self.d_out_keys_cccl = cccl.to_cccl_iter(d_out_keys)
        self.d_out_items_cccl = cccl.to_cccl_iter(d_out_items)
        self.d_out_num_selected_cccl = cccl.to_cccl_iter(d_out_num_selected)

        if isinstance(d_in_keys, IteratorBase):
            value_type = d_in_keys.value_type
        else:
            value_type = numba.from_dtype(protocols.get_dtype(d_in_keys))

        sig = (value_type, value_type)
        self.op_wrapper = cccl.to_cccl_op(op, sig)

        self.bindings = get_bindings()
        self.build_result = cccl.DeviceUniqueByKeyBuildResult()
        error = call_build(
            self.bindings.cccl_device_unique_by_key_build,
            ctypes.byref(self.build_result),
            self.d_in_keys_cccl,
            self.d_in_items_cccl,
            self.d_out_keys_cccl,
            self.d_out_items_cccl,
            self.d_out_num_selected_cccl,
            self.op_wrapper,
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error building unique_by_key")

    def __call__(
        self,
        temp_storage,
        d_in_keys: DeviceArrayLike | IteratorBase,
        d_in_items: DeviceArrayLike | IteratorBase,
        d_out_keys: DeviceArrayLike | IteratorBase,
        d_out_items: DeviceArrayLike | IteratorBase,
        d_out_num_selected: DeviceArrayLike,
        num_items: int,
        stream=None,
    ):
        set_state_fn = cccl.set_cccl_iterator_state
        set_state_fn(self.d_in_keys_cccl, d_in_keys)
        set_state_fn(self.d_in_items_cccl, d_in_items)
        set_state_fn(self.d_out_keys_cccl, d_out_keys)
        set_state_fn(self.d_out_items_cccl, d_out_items)
        set_state_fn(self.d_out_num_selected_cccl, d_out_num_selected)

        stream_handle = protocols.validate_and_get_stream(stream)
        if temp_storage is None:
            temp_storage_bytes = ctypes.c_size_t()
            d_temp_storage = None
        else:
            temp_storage_bytes = ctypes.c_size_t(temp_storage.nbytes)
            # Note: this is slightly slower, but supports all ndarray-like objects as long as they support CAI
            # TODO: switch to use gpumemoryview once it's ready
            d_temp_storage = temp_storage.__cuda_array_interface__["data"][0]

        error = self.bindings.cccl_device_unique_by_key(
            self.build_result,
            ctypes.c_void_p(d_temp_storage),
            ctypes.byref(temp_storage_bytes),
            self.d_in_keys_cccl,
            self.d_in_items_cccl,
            self.d_out_keys_cccl,
            self.d_out_items_cccl,
            self.d_out_num_selected_cccl,
            self.op_wrapper,
            ctypes.c_ulonglong(num_items),
            ctypes.c_void_p(stream_handle),
        )

        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error in unique by key")

        return temp_storage_bytes.value

    def __del__(self):
        if self.build_result is None:
            return
        self.bindings.cccl_device_unique_by_key_cleanup(ctypes.byref(self.build_result))


@cache_with_key(make_cache_key)
def unique_by_key(
    d_in_keys: DeviceArrayLike | IteratorBase,
    d_in_items: DeviceArrayLike | IteratorBase,
    d_out_keys: DeviceArrayLike | IteratorBase,
    d_out_items: DeviceArrayLike | IteratorBase,
    d_out_num_selected: DeviceArrayLike,
    op: Callable,
):
    """Implements a device-wide unique by key operation using ``d_in_keys`` and the comparison operator ``op``. Only the first key and its value from each run is selected and the total number of items selected is also reported.

    Example:
        Below, ``unique_by_key`` is used to populate the arrays of output keys and items with the first key and its corresponding item from each sequence of equal keys. It also outputs the number of items selected.

        .. literalinclude:: ../../python/cuda_parallel/tests/test_unique_by_key_api.py
          :language: python
          :dedent:
          :start-after: example-begin unique-by-key
          :end-before: example-end unique-by-key

    Args:
        d_in_keys: Device array or iterator containing the input sequence of keys
        d_in_items: Device array or iterator that contains each key's corresponding item
        d_out_keys: Device array or iterator to store the outputted keys
        d_out_items: Device array or iterator to store each outputted key's item
        d_out_num_selected: Device array to store how many items were selected
        op: Callable representing the equality operator

    Returns:
        A callable object that can be used to perform unique by key
    """

    return _UniqueByKey(
        d_in_keys, d_in_items, d_out_keys, d_out_items, d_out_num_selected, op
    )
