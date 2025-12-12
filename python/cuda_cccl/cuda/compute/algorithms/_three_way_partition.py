# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

import numba

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import cache_with_key
from .._cccl_interop import call_build, set_cccl_iterator_state
from .._utils import protocols
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..iterators._iterators import IteratorBase
from ..op import OpAdapter, make_op_adapter
from ..typing import DeviceArrayLike


def _make_cache_key(
    d_in: DeviceArrayLike | IteratorBase,
    d_first_part_out: DeviceArrayLike | IteratorBase,
    d_second_part_out: DeviceArrayLike | IteratorBase,
    d_unselected_out: DeviceArrayLike | IteratorBase,
    d_num_selected_out: DeviceArrayLike | IteratorBase,
    select_first_part_op: OpAdapter,
    select_second_part_op: OpAdapter,
):
    d_in_key = (
        d_in.kind if isinstance(d_in, IteratorBase) else protocols.get_dtype(d_in)
    )
    d_first_part_out_key = (
        d_first_part_out.kind
        if isinstance(d_first_part_out, IteratorBase)
        else protocols.get_dtype(d_first_part_out)
    )
    d_second_part_out_key = (
        d_second_part_out.kind
        if isinstance(d_second_part_out, IteratorBase)
        else protocols.get_dtype(d_second_part_out)
    )
    d_unselected_out_key = (
        d_unselected_out.kind
        if isinstance(d_unselected_out, IteratorBase)
        else protocols.get_dtype(d_unselected_out)
    )
    d_num_selected_out_key = (
        d_num_selected_out.kind
        if isinstance(d_num_selected_out, IteratorBase)
        else protocols.get_dtype(d_num_selected_out)
    )

    return (
        d_in_key,
        d_first_part_out_key,
        d_second_part_out_key,
        d_unselected_out_key,
        d_num_selected_out_key,
        select_first_part_op.get_cache_key(),
        select_second_part_op.get_cache_key(),
    )


class _ThreeWayPartition:
    __slots__ = [
        "build_result",
        "d_in_cccl",
        "d_first_part_out_cccl",
        "d_second_part_out_cccl",
        "d_unselected_out_cccl",
        "d_num_selected_out_cccl",
        "select_first_part_op",
        "select_second_part_op",
        "select_first_part_op_cccl",
        "select_second_part_op_cccl",
    ]

    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_first_part_out: DeviceArrayLike | IteratorBase,
        d_second_part_out: DeviceArrayLike | IteratorBase,
        d_unselected_out: DeviceArrayLike | IteratorBase,
        d_num_selected_out: DeviceArrayLike | IteratorBase,
        select_first_part_op: OpAdapter,
        select_second_part_op: OpAdapter,
    ):
        self.d_in_cccl = cccl.to_cccl_input_iter(d_in)
        self.d_first_part_out_cccl = cccl.to_cccl_output_iter(d_first_part_out)
        self.d_second_part_out_cccl = cccl.to_cccl_output_iter(d_second_part_out)
        self.d_unselected_out_cccl = cccl.to_cccl_output_iter(d_unselected_out)
        self.d_num_selected_out_cccl = cccl.to_cccl_output_iter(d_num_selected_out)

        # Compile ops - partition predicates return uint8 (boolean)
        value_type = cccl.get_value_type(d_in)
        self.select_first_part_op_cccl = select_first_part_op.compile(
            (value_type,), numba.types.uint8
        )
        self.select_second_part_op_cccl = select_second_part_op.compile(
            (value_type,), numba.types.uint8
        )

        self.build_result = call_build(
            _bindings.DeviceThreeWayPartitionBuildResult,
            self.d_in_cccl,
            self.d_first_part_out_cccl,
            self.d_second_part_out_cccl,
            self.d_unselected_out_cccl,
            self.d_num_selected_out_cccl,
            self.select_first_part_op_cccl,
            self.select_second_part_op_cccl,
        )

    def __call__(
        self,
        temp_storage,
        d_in,
        d_first_part_out,
        d_second_part_out,
        d_unselected_out,
        d_num_selected_out,
        num_items: int,
        stream=None,
    ):
        set_cccl_iterator_state(self.d_in_cccl, d_in)
        set_cccl_iterator_state(self.d_first_part_out_cccl, d_first_part_out)
        set_cccl_iterator_state(self.d_second_part_out_cccl, d_second_part_out)
        set_cccl_iterator_state(self.d_unselected_out_cccl, d_unselected_out)
        set_cccl_iterator_state(self.d_num_selected_out_cccl, d_num_selected_out)

        stream_handle = protocols.validate_and_get_stream(stream)

        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            d_temp_storage = protocols.get_data_pointer(temp_storage)

        temp_storage_bytes = self.build_result.compute(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_cccl,
            self.d_first_part_out_cccl,
            self.d_second_part_out_cccl,
            self.d_unselected_out_cccl,
            self.d_num_selected_out_cccl,
            self.select_first_part_op_cccl,
            self.select_second_part_op_cccl,
            num_items,
            stream_handle,
        )
        return temp_storage_bytes


@cache_with_key(_make_cache_key)
def _make_three_way_partition_cached(
    d_in: DeviceArrayLike | IteratorBase,
    d_first_part_out: DeviceArrayLike | IteratorBase,
    d_second_part_out: DeviceArrayLike | IteratorBase,
    d_unselected_out: DeviceArrayLike | IteratorBase,
    d_num_selected_out: DeviceArrayLike | IteratorBase,
    select_first_part_op: OpAdapter,
    select_second_part_op: OpAdapter,
):
    """Internal cached factory for _ThreeWayPartition."""
    return _ThreeWayPartition(
        d_in,
        d_first_part_out,
        d_second_part_out,
        d_unselected_out,
        d_num_selected_out,
        select_first_part_op,
        select_second_part_op,
    )


def make_three_way_partition(
    d_in: DeviceArrayLike | IteratorBase,
    d_first_part_out: DeviceArrayLike | IteratorBase,
    d_second_part_out: DeviceArrayLike | IteratorBase,
    d_unselected_out: DeviceArrayLike | IteratorBase,
    d_num_selected_out: DeviceArrayLike | IteratorBase,
    select_first_part_op: Callable | OpAdapter,
    select_second_part_op: Callable | OpAdapter,
):
    """
    Computes a device-wide three-way partition using the specified unary ``select_first_part_op`` and ``select_second_part_op`` operators.

    Example:
        Below, ``make_three_way_partition`` is used to create a three-way partition object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/partition/three_way_partition_object.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_first_part_out: Device array or iterator to store the first part of the output
        d_second_part_out: Device array or iterator to store the second part of the output
        d_unselected_out: Device array or iterator to store the unselected items
        d_num_selected_out: Device array to store the number of items selected. The total number of items selected by ``select_first_part_op`` and ``select_second_part_op`` is stored in ``d_num_selected_out[0]`` and ``d_num_selected_out[1]``, respectively.
        select_first_part_op: Callable representing the unary operator to select the first part.
            Can reference device arrays as globals/closures - they will be automatically captured.
        select_second_part_op: Callable representing the unary operator to select the second part.
            Can reference device arrays as globals/closures - they will be automatically captured.

    Returns:
        A callable object that can be used to perform the three-way partition
    """
    first_op_adapter = make_op_adapter(select_first_part_op)
    second_op_adapter = make_op_adapter(select_second_part_op)

    return _make_three_way_partition_cached(
        d_in,
        d_first_part_out,
        d_second_part_out,
        d_unselected_out,
        d_num_selected_out,
        first_op_adapter,
        second_op_adapter,
    )


def three_way_partition(
    d_in: DeviceArrayLike | IteratorBase,
    d_first_part_out: DeviceArrayLike | IteratorBase,
    d_second_part_out: DeviceArrayLike | IteratorBase,
    d_unselected_out: DeviceArrayLike | IteratorBase,
    d_num_selected_out: DeviceArrayLike | IteratorBase,
    select_first_part_op: Callable,
    select_second_part_op: Callable,
    num_items: int,
    stream=None,
):
    """
    Performs device-wide three-way partition. Given an input sequence of data items, it partitions the items into three parts:
    - The first part is selected by the ``select_first_part_op`` operator.
    - The second part is selected by the ``select_second_part_op`` operator.
    - The unselected items are not selected by either operator.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``three_way_partition`` is used to partition a sequence of integers into three parts.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/partition/three_way_partition_basic.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_first_part_out: Device array or iterator to store the first part of the output
        d_second_part_out: Device array or iterator to store the second part of the output
        d_unselected_out: Device array or iterator to store the unselected items
        d_num_selected_out: Device array to store the number of items selected. The total number of items selected by ``select_first_part_op`` and ``select_second_part_op`` is stored in ``d_num_selected_out[0]`` and ``d_num_selected_out[1]``, respectively.
        select_first_part_op: Callable representing the unary operator to select the first part
        select_second_part_op: Callable representing the unary operator to select the second part
        num_items: Number of items to partition
        stream: CUDA stream for the operation (optional)
    """
    partitioner = make_three_way_partition(
        d_in,
        d_first_part_out,
        d_second_part_out,
        d_unselected_out,
        d_num_selected_out,
        select_first_part_op,
        select_second_part_op,
    )
    tmp_storage_bytes = partitioner(
        None,
        d_in,
        d_first_part_out,
        d_second_part_out,
        d_unselected_out,
        d_num_selected_out,
        num_items,
        stream,
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    partitioner(
        tmp_storage,
        d_in,
        d_first_part_out,
        d_second_part_out,
        d_unselected_out,
        d_num_selected_out,
        num_items,
        stream,
    )
