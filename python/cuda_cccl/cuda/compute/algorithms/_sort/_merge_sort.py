# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

from ... import _bindings, types
from ... import _cccl_interop as cccl
from ..._caching import cache_with_registered_key_functions
from ..._cccl_interop import call_build, set_cccl_iterator_state
from ..._utils.protocols import (
    get_data_pointer,
    validate_and_get_stream,
)
from ..._utils.temp_storage_buffer import TempStorageBuffer
from ...iterators._iterators import IteratorBase
from ...op import OpAdapter, OpKind, make_op_adapter
from ...typing import DeviceArrayLike


class _MergeSort:
    __slots__ = [
        "d_in_keys_cccl",
        "d_in_items_cccl",
        "d_out_keys_cccl",
        "d_out_items_cccl",
        "op_adapter",
        "op_cccl",
        "build_result",
    ]

    def __init__(
        self,
        d_in_keys: DeviceArrayLike | IteratorBase,
        d_in_items: DeviceArrayLike | IteratorBase | None,
        d_out_keys: DeviceArrayLike,
        d_out_items: DeviceArrayLike | None,
        op: OpAdapter,
    ):
        present_in_values = d_in_items is not None
        present_out_values = d_out_items is not None
        assert present_in_values == present_out_values

        self.d_in_keys_cccl = cccl.to_cccl_input_iter(d_in_keys)
        self.d_in_items_cccl = cccl.to_cccl_input_iter(d_in_items)
        self.d_out_keys_cccl = cccl.to_cccl_output_iter(d_out_keys)
        self.d_out_items_cccl = cccl.to_cccl_output_iter(d_out_items)
        self.op_adapter = op

        # Compile the op - merge_sort expects int8 return (comparison)
        value_type = cccl.get_value_type(d_in_keys)
        self.op_cccl = op.compile((value_type, value_type), types.int8)

        self.build_result = call_build(
            _bindings.DeviceMergeSortBuildResult,
            self.d_in_keys_cccl,
            self.d_in_items_cccl,
            self.d_out_keys_cccl,
            self.d_out_items_cccl,
            self.op_cccl,
        )

    def __call__(
        self,
        temp_storage,
        d_in_keys: DeviceArrayLike | IteratorBase,
        d_in_items: DeviceArrayLike | IteratorBase | None,
        d_out_keys: DeviceArrayLike,
        d_out_items: DeviceArrayLike | None,
        op: Callable | OpKind | OpAdapter,
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

        # Update op state for stateful ops
        op_adapter = make_op_adapter(op)
        op_adapter.update_op_state(self.op_cccl)

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
            self.op_cccl,
            stream_handle,
        )

        return temp_storage_bytes


@cache_with_registered_key_functions
def make_merge_sort(
    d_in_keys: DeviceArrayLike | IteratorBase,
    d_in_items: DeviceArrayLike | IteratorBase | None,
    d_out_keys: DeviceArrayLike,
    d_out_items: DeviceArrayLike | None,
    op: Callable | OpKind,
):
    """Implements a device-wide merge sort using ``d_in_keys`` and the comparison operator ``op``.

    Example:
        Below, ``make_merge_sort`` is used to create a merge sort object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/sort/merge_sort_object.py
          :language: python
          :start-after: # example-begin


    Args:
        d_in_keys: Device array or iterator containing the input keys to be sorted
        d_in_items: Optional device array or iterator that contains each key's corresponding item
        d_out_keys: Device array to store the sorted keys
        d_out_items: Device array to store the sorted items
        op: Callable or OpKind representing the comparison operator

    Returns:
        A callable object that can be used to perform the merge sort
    """
    op_adapter = make_op_adapter(op)
    return _MergeSort(d_in_keys, d_in_items, d_out_keys, d_out_items, op_adapter)


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

    Example:
        Below, ``merge_sort`` is used to sort a sequence of keys inplace. It also rearranges the items according to the keys' order.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/sort/merge_sort_basic.py
            :language: python
            :start-after: # example-begin


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
        None, d_in_keys, d_in_items, d_out_keys, d_out_items, op, num_items, stream
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    sorter(
        tmp_storage,
        d_in_keys,
        d_in_items,
        d_out_keys,
        d_out_items,
        op,
        num_items,
        stream,
    )
