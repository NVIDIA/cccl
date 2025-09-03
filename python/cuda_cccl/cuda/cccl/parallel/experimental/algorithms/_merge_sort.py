# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Union

import numba

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import CachableFunction, cache_with_key
from .._cccl_interop import call_build, set_cccl_iterator_state
from .._utils import protocols
from .._utils.protocols import (
    get_data_pointer,
    validate_and_get_stream,
)
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..iterators._iterators import IteratorBase
from ..op import OpKind
from ..typing import DeviceArrayLike


def make_cache_key(
    d_in_keys: DeviceArrayLike | IteratorBase,
    d_in_items: DeviceArrayLike | IteratorBase | None,
    d_out_keys: DeviceArrayLike,
    d_out_items: DeviceArrayLike | None,
    op: Callable | OpKind,
):
    d_in_keys_key = (
        d_in_keys.kind
        if isinstance(d_in_keys, IteratorBase)
        else protocols.get_dtype(d_in_keys)
    )
    if d_in_items is None:
        d_in_items_key = None
    else:
        d_in_items_key = (
            d_in_items.kind
            if isinstance(d_in_items, IteratorBase)
            else protocols.get_dtype(d_in_items)
        )
    d_out_keys_key = protocols.get_dtype(d_out_keys)
    if d_out_items is None:
        d_out_items_key = None
    else:
        d_out_items_key = (
            d_out_items.kind
            if isinstance(d_out_items, IteratorBase)
            else protocols.get_dtype(d_out_items)
        )

    # Handle well-known operations differently
    op_key: Union[tuple[str, int], CachableFunction]
    if isinstance(op, OpKind):
        op_key = (op.name, op.value)
    else:
        op_key = CachableFunction(op)

    return (d_in_keys_key, d_in_items_key, d_out_keys_key, d_out_items_key, op_key)


class _MergeSort:
    __slots__ = [
        "d_in_keys_cccl",
        "d_in_items_cccl",
        "d_out_keys_cccl",
        "d_out_items_cccl",
        "op_wrapper",
        "build_result",
    ]

    def __init__(
        self,
        d_in_keys: DeviceArrayLike | IteratorBase,
        d_in_items: DeviceArrayLike | IteratorBase | None,
        d_out_keys: DeviceArrayLike,
        d_out_items: DeviceArrayLike | None,
        op: Callable | OpKind,
    ):
        present_in_values = d_in_items is not None
        present_out_values = d_out_items is not None
        assert present_in_values == present_out_values

        self.d_in_keys_cccl = cccl.to_cccl_iter(d_in_keys)
        self.d_in_items_cccl = cccl.to_cccl_iter(d_in_items)
        self.d_out_keys_cccl = cccl.to_cccl_iter(d_out_keys)
        self.d_out_items_cccl = cccl.to_cccl_iter(d_out_items)

        value_type = cccl.get_value_type(d_in_keys)

        # For well-known operations, we don't need a signature
        if isinstance(op, OpKind):
            self.op_wrapper = cccl.to_cccl_op(op, None)
        else:
            sig = numba.types.int8(value_type, value_type)
            self.op_wrapper = cccl.to_cccl_op(op, sig)

        self.build_result = call_build(
            _bindings.DeviceMergeSortBuildResult,
            self.d_in_keys_cccl,
            self.d_in_items_cccl,
            self.d_out_keys_cccl,
            self.d_out_items_cccl,
            self.op_wrapper,
        )

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
        present_in_values = d_in_items is not None
        present_out_values = d_out_items is not None
        assert present_in_values == present_out_values

        set_cccl_iterator_state(self.d_in_keys_cccl, d_in_keys)
        if present_in_values:
            set_cccl_iterator_state(self.d_in_items_cccl, d_in_items)
        set_cccl_iterator_state(self.d_out_keys_cccl, d_out_keys)
        if present_out_values:
            set_cccl_iterator_state(self.d_out_items_cccl, d_out_items)

        stream_handle = validate_and_get_stream(stream)
        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            # Note: this is slightly slower, but supports all ndarray-like objects as long as they support CAI
            # TODO: switch to use gpumemoryview once it's ready
            d_temp_storage = get_data_pointer(temp_storage)

        temp_storage_bytes = self.build_result.compute(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_keys_cccl,
            self.d_in_items_cccl,
            self.d_out_keys_cccl,
            self.d_out_items_cccl,
            num_items,
            self.op_wrapper,
            stream_handle,
        )

        return temp_storage_bytes


@cache_with_key(make_cache_key)
def make_merge_sort(
    d_in_keys: DeviceArrayLike | IteratorBase,
    d_in_items: DeviceArrayLike | IteratorBase | None,
    d_out_keys: DeviceArrayLike,
    d_out_items: DeviceArrayLike | None,
    op: Callable | OpKind,
):
    """Implements a device-wide merge sort using ``d_in_keys`` and the comparison operator ``op``.

    Example:
        Below, ``merge_sort`` is used to sort a sequence of keys inplace. It also rearranges the items according to the keys' order.

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_merge_sort_api.py
          :language: python
          :dedent:
          :start-after: example-begin merge-sort
          :end-before: example-end merge-sort

    Args:
        d_in_keys: Device array or iterator containing the input keys to be sorted
        d_in_items: Optional device array or iterator that contains each key's corresponding item
        d_out_keys: Device array to store the sorted keys
        d_out_items: Device array to store the sorted items
        op: Callable or OpKind representing the comparison operator

    Returns:
        A callable object that can be used to perform the merge sort
    """
    return _MergeSort(d_in_keys, d_in_items, d_out_keys, d_out_items, op)


def merge_sort(
    d_in_keys: DeviceArrayLike | IteratorBase,
    d_in_items: DeviceArrayLike | IteratorBase | None,
    d_out_keys: DeviceArrayLike,
    d_out_items: DeviceArrayLike | None,
    op: Callable | OpKind,
    num_items: int,
    stream=None,
):
    """
    Performs device-wide merge sort.

    This function automatically handles temporary storage allocation and execution.

    Args:
        d_in_keys: Device array or iterator containing the input sequence of keys
        d_in_items: Device array or iterator containing the input sequence of items (optional)
        d_out_keys: Device array to store the sorted keys
        d_out_items: Device array to store the sorted items (optional)
        op: Comparison operator for sorting
        num_items: Number of items to sort
        stream: CUDA stream for the operation (optional)
    """
    sorter = make_merge_sort(d_in_keys, d_in_items, d_out_keys, d_out_items, op)
    tmp_storage_bytes = sorter(
        None, d_in_keys, d_in_items, d_out_keys, d_out_items, num_items, stream
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    sorter(
        tmp_storage, d_in_keys, d_in_items, d_out_keys, d_out_items, num_items, stream
    )
