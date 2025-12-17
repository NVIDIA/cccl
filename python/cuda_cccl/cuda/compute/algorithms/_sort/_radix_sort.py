# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ... import _bindings
from ... import _cccl_interop as cccl
from ..._caching import cache_with_key
from ..._cccl_interop import call_build, set_cccl_iterator_state
from ..._utils.protocols import (
    get_data_pointer,
    get_dtype,
    validate_and_get_stream,
)
from ..._utils.temp_storage_buffer import TempStorageBuffer
from ...typing import DeviceArrayLike
from ._sort_common import DoubleBuffer, SortOrder, _get_arrays


def make_cache_key(
    d_in_keys: DeviceArrayLike | DoubleBuffer,
    d_out_keys: DeviceArrayLike | None,
    d_in_values: DeviceArrayLike | DoubleBuffer | None,
    d_out_values: DeviceArrayLike | None,
    order: SortOrder,
):
    d_in_keys_array, d_out_keys_array, d_in_values_array, d_out_values_array = (
        _get_arrays(d_in_keys, d_out_keys, d_in_values, d_out_values)
    )

    d_in_keys_key = get_dtype(d_in_keys_array)
    d_in_values_key = (
        None if d_in_values_array is None else get_dtype(d_in_values_array)
    )
    d_out_keys_key = get_dtype(d_out_keys_array)
    d_out_values_key = (
        None if d_out_values_array is None else get_dtype(d_out_values_array)
    )

    return (
        d_in_keys_key,
        d_out_keys_key,
        d_in_values_key,
        d_out_values_key,
        order,
    )


class _RadixSort:
    __slots__ = [
        "d_in_keys_cccl",
        "d_out_keys_cccl",
        "d_in_values_cccl",
        "d_out_values_cccl",
        "decomposer_op",
        "build_result",
    ]

    def __init__(
        self,
        d_in_keys: DeviceArrayLike | DoubleBuffer,
        d_out_keys: DeviceArrayLike | None,
        d_in_values: DeviceArrayLike | DoubleBuffer | None,
        d_out_values: DeviceArrayLike | None,
        order: SortOrder,
    ):
        d_in_keys_array, d_out_keys_array, d_in_values_array, d_out_values_array = (
            _get_arrays(d_in_keys, d_out_keys, d_in_values, d_out_values)
        )

        self.d_in_keys_cccl = cccl.to_cccl_input_iter(d_in_keys_array)
        self.d_out_keys_cccl = cccl.to_cccl_output_iter(d_out_keys_array)
        self.d_in_values_cccl = cccl.to_cccl_input_iter(d_in_values_array)
        self.d_out_values_cccl = cccl.to_cccl_output_iter(d_out_values_array)

        # TODO: decomposer op is not supported for now
        self.decomposer_op = cccl.Op(
            name="",
            operator_type=cccl.OpKind.STATELESS,
            ltoir=b"",
            state_alignment=1,
            state=None,
        )
        decomposer_return_type = "".encode("utf-8")

        self.build_result = call_build(
            _bindings.DeviceRadixSortBuildResult,
            _bindings.SortOrder.ASCENDING
            if order is SortOrder.ASCENDING
            else _bindings.SortOrder.DESCENDING,
            self.d_in_keys_cccl,
            self.d_in_values_cccl,
            self.decomposer_op,
            decomposer_return_type,
        )

    def __call__(
        self,
        temp_storage,
        d_in_keys: DeviceArrayLike | DoubleBuffer,
        d_out_keys: DeviceArrayLike | None,
        d_in_values: DeviceArrayLike | DoubleBuffer | None,
        d_out_values: DeviceArrayLike | None,
        num_items: int,
        begin_bit: int | None = None,
        end_bit: int | None = None,
        stream=None,
    ):
        d_in_keys_array, d_out_keys_array, d_in_values_array, d_out_values_array = (
            _get_arrays(d_in_keys, d_out_keys, d_in_values, d_out_values)
        )

        set_cccl_iterator_state(self.d_in_keys_cccl, d_in_keys_array)
        if d_in_values_array is not None:
            set_cccl_iterator_state(self.d_in_values_cccl, d_in_values_array)
        set_cccl_iterator_state(self.d_out_keys_cccl, d_out_keys_array)
        if d_out_values_array is not None:
            set_cccl_iterator_state(self.d_out_values_cccl, d_out_values_array)

        is_overwrite_okay = isinstance(d_in_keys, DoubleBuffer)

        stream_handle = validate_and_get_stream(stream)
        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            # Note: this is slightly slower, but supports all ndarray-like objects as long as they support CAI
            # TODO: switch to use gpumemoryview once it's ready
            d_temp_storage = get_data_pointer(temp_storage)

        if begin_bit is None:
            begin_bit = 0
        if end_bit is None:
            key_type = get_dtype(d_in_keys_array)
            end_bit = key_type.itemsize * 8

        selector = -1

        temp_storage_bytes, selector = self.build_result.compute(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_keys_cccl,
            self.d_out_keys_cccl,
            self.d_in_values_cccl,
            self.d_out_values_cccl,
            self.decomposer_op,
            num_items,
            begin_bit,
            end_bit,
            is_overwrite_okay,
            selector,
            stream_handle,
        )

        if is_overwrite_okay and temp_storage is not None:
            assert selector in (0, 1)
            assert isinstance(d_in_keys, DoubleBuffer)
            d_in_keys.selector = selector
            if d_in_values is not None:
                assert isinstance(d_in_values, DoubleBuffer)
                d_in_values.selector = selector

        return temp_storage_bytes


@cache_with_key(make_cache_key)
def make_radix_sort(
    d_in_keys: DeviceArrayLike | DoubleBuffer,
    d_out_keys: DeviceArrayLike | None,
    d_in_values: DeviceArrayLike | DoubleBuffer | None,
    d_out_values: DeviceArrayLike | None,
    order: SortOrder,
):
    """Implements a device-wide radix sort using ``d_in_keys`` in the requested order.

    Example:
        Below, ``make_radix_sort`` is used to create a radix sort object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/sort/radix_sort_object.py
          :language: python
          :start-after: # example-begin


    Args:
        d_in_keys: Device array or DoubleBuffer containing the input keys to be sorted
        d_out_keys: Device array to store the sorted keys
        d_in_values: Optional Device array or DoubleBuffer containing the input keys to be sorted
        d_out_values: Device array to store the sorted values
        op: Callable representing the comparison operator

    Returns:
        A callable object that can be used to perform the radix sort
    """
    return _RadixSort(d_in_keys, d_out_keys, d_in_values, d_out_values, order)


def radix_sort(
    d_in_keys: DeviceArrayLike | DoubleBuffer,
    d_out_keys: DeviceArrayLike | None,
    d_in_values: DeviceArrayLike | DoubleBuffer | None,
    d_out_values: DeviceArrayLike | None,
    order: SortOrder,
    num_items: int,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    stream=None,
):
    """
    Performs device-wide radix sort.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``radix_sort`` is used to sort a sequence of keys. It also rearranges the values according to the keys' order.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/sort/radix_sort_basic.py
            :language: python
            :start-after: # example-begin


        In the following example, ``radix_sort`` is used to sort a sequence of keys with a ``DoubleBuffer` for reduced temporary storage.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/sort/radix_sort_buffer.py
            :language: python
            :start-after: # example-begin


    Args:
        d_in_keys: Device array or DoubleBuffer containing the input sequence of keys
        d_out_keys: Device array to store the sorted keys (optional)
        d_in_values: Device array or DoubleBuffer containing the input sequence of values (optional)
        d_out_values: Device array to store the sorted values (optional)
        order: Sort order (ascending or descending)
        num_items: Number of items to sort
        begin_bit: Beginning bit position for comparison (optional)
        end_bit: Ending bit position for comparison (optional)
        stream: CUDA stream for the operation (optional)
    """
    sorter = make_radix_sort(d_in_keys, d_out_keys, d_in_values, d_out_values, order)
    tmp_storage_bytes = sorter(
        None,
        d_in_keys,
        d_out_keys,
        d_in_values,
        d_out_values,
        num_items,
        begin_bit,
        end_bit,
        stream,
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    sorter(
        tmp_storage,
        d_in_keys,
        d_out_keys,
        d_in_values,
        d_out_values,
        num_items,
        begin_bit,
        end_bit,
        stream,
    )
