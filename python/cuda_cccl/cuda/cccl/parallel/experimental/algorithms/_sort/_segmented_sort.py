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


class _SegmentedSort:
    __slots__ = [
        "build_result",
        "d_in_keys_cccl",
        "d_out_keys_cccl",
        "d_in_values_cccl",
        "d_out_values_cccl",
        "start_offsets_in_cccl",
        "end_offsets_in_cccl",
    ]

    def __init__(
        self,
        d_in_keys: DeviceArrayLike | DoubleBuffer,
        d_out_keys: DeviceArrayLike | None,
        d_in_values: DeviceArrayLike | DoubleBuffer | None,
        d_out_values: DeviceArrayLike | None,
        start_offsets_in: DeviceArrayLike,
        end_offsets_in: DeviceArrayLike,
        order: SortOrder,
    ):
        d_in_keys_array, d_out_keys_array, d_in_values_array, d_out_values_array = (
            _get_arrays(d_in_keys, d_out_keys, d_in_values, d_out_values)
        )

        self.d_in_keys_cccl = cccl.to_cccl_input_iter(d_in_keys_array)
        self.d_out_keys_cccl = cccl.to_cccl_output_iter(d_out_keys_array)
        self.d_in_values_cccl = cccl.to_cccl_input_iter(d_in_values_array)
        self.d_out_values_cccl = cccl.to_cccl_output_iter(d_out_values_array)
        self.start_offsets_in_cccl = cccl.to_cccl_input_iter(start_offsets_in)
        self.end_offsets_in_cccl = cccl.to_cccl_input_iter(end_offsets_in)

        cccl.cccl_iterator_set_host_advance(
            self.start_offsets_in_cccl, start_offsets_in
        )
        cccl.cccl_iterator_set_host_advance(self.end_offsets_in_cccl, end_offsets_in)

        self.build_result = call_build(
            _bindings.DeviceSegmentedSortBuildResult,
            _bindings.SortOrder.ASCENDING
            if order is SortOrder.ASCENDING
            else _bindings.SortOrder.DESCENDING,
            self.d_in_keys_cccl,
            self.d_in_values_cccl,
            self.start_offsets_in_cccl,
            self.end_offsets_in_cccl,
        )

    def __call__(
        self,
        temp_storage,
        d_in_keys,
        d_out_keys,
        d_in_values,
        d_out_values,
        num_items,
        num_segments,
        start_offsets_in,
        end_offsets_in,
        stream=None,
    ):
        d_in_keys_array, d_out_keys_array, d_in_values_array, d_out_values_array = (
            _get_arrays(d_in_keys, d_out_keys, d_in_values, d_out_values)
        )

        set_cccl_iterator_state(self.d_in_keys_cccl, d_in_keys_array)
        set_cccl_iterator_state(self.d_out_keys_cccl, d_out_keys_array)
        if d_in_values_array is not None:
            set_cccl_iterator_state(self.d_in_values_cccl, d_in_values_array)
        if d_out_values_array is not None:
            set_cccl_iterator_state(self.d_out_values_cccl, d_out_values_array)
        set_cccl_iterator_state(self.start_offsets_in_cccl, start_offsets_in)
        set_cccl_iterator_state(self.end_offsets_in_cccl, end_offsets_in)

        stream_handle = validate_and_get_stream(stream)
        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            d_temp_storage = get_data_pointer(temp_storage)

        # Detect overwrite mode and selector, similar to radix sort
        is_overwrite_okay = isinstance(d_in_keys, DoubleBuffer)
        selector = -1

        temp_storage_bytes, selector = self.build_result.compute(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_keys_cccl,
            self.d_out_keys_cccl,
            self.d_in_values_cccl,
            self.d_out_values_cccl,
            num_items,
            num_segments,
            self.start_offsets_in_cccl,
            self.end_offsets_in_cccl,
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


def make_cache_key(
    d_in_keys: DeviceArrayLike | DoubleBuffer,
    d_out_keys: DeviceArrayLike | None,
    d_in_values: DeviceArrayLike | DoubleBuffer | None,
    d_out_values: DeviceArrayLike | None,
    start_offsets_in: DeviceArrayLike,
    end_offsets_in: DeviceArrayLike,
    order: SortOrder,
):
    d_in_keys_array, d_out_keys_array, d_in_values_array, d_out_values_array = (
        _get_arrays(d_in_keys, d_out_keys, d_in_values, d_out_values)
    )

    d_in_keys_key = get_dtype(d_in_keys_array)
    d_out_keys_key = None if d_out_keys_array is None else get_dtype(d_out_keys_array)
    d_in_values_key = (
        None if d_in_values_array is None else get_dtype(d_in_values_array)
    )
    d_out_values_key = (
        None if d_out_values_array is None else get_dtype(d_out_values_array)
    )
    start_offsets_in_key = get_dtype(start_offsets_in)
    end_offsets_in_key = get_dtype(end_offsets_in)

    return (
        d_in_keys_key,
        d_out_keys_key,
        d_in_values_key,
        d_out_values_key,
        start_offsets_in_key,
        end_offsets_in_key,
        order,
    )


@cache_with_key(make_cache_key)
def make_segmented_sort(
    d_in_keys: DeviceArrayLike | DoubleBuffer,
    d_out_keys: DeviceArrayLike | None,
    d_in_values: DeviceArrayLike | DoubleBuffer | None,
    d_out_values: DeviceArrayLike | None,
    start_offsets_in: DeviceArrayLike,
    end_offsets_in: DeviceArrayLike,
    order: SortOrder,
):
    """
    Performs a device-wide segmented sort using the specified keys and values.

    Example:
        Below, ``make_segmented_sort`` is used to create a segmented sort object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/sort/segmented_sort_object.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in_keys: Device array or DoubleBuffer containing the input keys to be sorted
        d_out_keys: Device array to store the sorted keys
        d_in_values: Optional Device array or DoubleBuffer containing the input values to be sorted
        d_out_values: Device array to store the sorted values
        start_offsets_in: Device array or iterator containing the sequence of beginning offsets
        end_offsets_in: Device array or iterator containing the sequence of ending offsets
        order: SortOrder specifying the order of the sort

    Returns:
        A callable object that can be used to perform the segmented sort
    """
    return _SegmentedSort(
        d_in_keys,
        d_out_keys,
        d_in_values,
        d_out_values,
        start_offsets_in,
        end_offsets_in,
        order,
    )


def segmented_sort(
    d_in_keys: DeviceArrayLike | DoubleBuffer,
    d_out_keys: DeviceArrayLike | None,
    d_in_values: DeviceArrayLike | DoubleBuffer | None,
    d_out_values: DeviceArrayLike | None,
    num_items: int,
    num_segments: int,
    start_offsets_in: DeviceArrayLike,
    end_offsets_in: DeviceArrayLike,
    order: SortOrder,
    stream=None,
):
    """
    Performs device-wide segmented sort.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``segmented_sort`` is used to perform a segmented sort. It also rearranges the values according to the keys' order.

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/sort/segmented_sort_basic.py
            :language: python
            :start-after: # example-begin


        In the following example, ``segmented_sort`` is used to perform a segmented sort with a ``DoubleBuffer` for reduced temporary storage.

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/sort/segmented_sort_buffer.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in_keys: Device array or DoubleBuffer containing the input keys to be sorted
        d_out_keys: Device array to store the sorted keys (optional)
        d_in_values: Device array or DoubleBuffer containing the input values to be sorted (optional)
        d_out_values: Device array to store the sorted values (optional)
        num_items: Total number of items to sort
        num_segments: Number of segments to sort
        start_offsets_in: Device array or iterator containing the sequence of beginning offsets
        end_offsets_in: Device array or iterator containing the sequence of ending offsets
        order: Sort order (ascending or descending)
        stream: CUDA stream for the operation (optional)
    """
    sorter = make_segmented_sort(
        d_in_keys,
        d_out_keys,
        d_in_values,
        d_out_values,
        start_offsets_in,
        end_offsets_in,
        order,
    )
    tmp_storage_bytes = sorter(
        None,
        d_in_keys,
        d_out_keys,
        d_in_values,
        d_out_values,
        num_items,
        num_segments,
        start_offsets_in,
        end_offsets_in,
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
        num_segments,
        start_offsets_in,
        end_offsets_in,
        stream,
    )
