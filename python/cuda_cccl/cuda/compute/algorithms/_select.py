# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

from .._caching import cache_with_registered_key_functions
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..iterators._factories import DiscardIterator
from ..iterators._iterators import IteratorBase
from ..op import OpAdapter, make_op_adapter
from ..typing import DeviceArrayLike
from ._three_way_partition import make_three_way_partition


class _Select:
    __slots__ = ["partitioner", "discard_second", "discard_unselected", "false_op"]

    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike | IteratorBase,
        d_num_selected_out: DeviceArrayLike,
        cond: OpAdapter,
    ):
        # Create discard iterators for unused outputs, using d_out as reference
        # to match the input/output type
        self.discard_second = DiscardIterator(d_out)
        self.discard_unselected = DiscardIterator(d_out)

        # Create adapter for the always-false second predicate
        self.false_op = make_op_adapter(lambda x: False)

        # Use three_way_partition internally
        self.partitioner = make_three_way_partition(
            d_in,
            d_out,  # first_part_out - this is where selected items go
            self.discard_second,  # second_part_out - discarded
            self.discard_unselected,  # unselected_out - discarded
            d_num_selected_out,
            cond,  # select_first_part_op - user's select condition
            self.false_op,  # select_second_part_op - always false
        )

    def __call__(
        self,
        temp_storage,
        d_in,
        d_out,
        d_num_selected_out,
        cond,
        num_items: int,
        stream=None,
    ):
        return self.partitioner(
            temp_storage,
            d_in,
            d_out,
            self.discard_second,
            self.discard_unselected,
            d_num_selected_out,
            make_op_adapter(cond),
            self.false_op,
            num_items,
            stream,
        )


@cache_with_registered_key_functions
def make_select(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    d_num_selected_out: DeviceArrayLike,
    cond: Callable,
):
    """
    Create a select object that can be called to select elements matching a condition.

    This is the object-oriented API that allows explicit control over temporary
    storage allocation. For simpler usage, consider using :func:`select`.

    Example:
        Below, ``make_select`` is used to create a select object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/select/select_object.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in: Device array or iterator containing the input sequence of data items.
        d_out: Device array or iterator to store the selected output items.
        d_num_selected_out: Device array to store the number of items that passed the selection.
            The count is stored in ``d_num_selected_out[0]``.
        cond: Callable representing the selection condition (predicate). Should return a
            boolean-like value (typically uint8) where non-zero means the item passes the selection.

    Returns:
        A callable object that performs the selection operation.
    """
    cond_adapter = make_op_adapter(cond)
    # Note: _Select internally calls make_three_way_partition which will
    # normalize the cond. But we've already normalized it, so the Op
    # will be passed through make_op unchanged.
    return _Select(d_in, d_out, d_num_selected_out, cond_adapter)


def select(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    d_num_selected_out: DeviceArrayLike,
    cond: Callable,
    num_items: int,
    stream=None,
):
    """
    Performs device-wide selection of elements based on a condition.

    Given an input sequence, this function selects all elements for which the condition
    function ``cond`` returns true (non-zero) and writes them to the output in a
    compacted form. The number of selected elements is written to ``d_num_selected_out[0]``.

    This function automatically handles temporary storage allocation and execution.

    The ``cond`` function can reference device arrays as globals or closures - they will
    be automatically captured as state arrays, enabling stateful operations like counting.

    Example:
        Below, ``select`` is used to select even numbers from an input array:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/select/select_basic.py
            :language: python
            :start-after: # example-begin

        You can also use iterators for more complex selection patterns:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/select/select_with_iterator.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in: Device array or iterator containing the input sequence of data items.
        d_out: Device array or iterator to store the selected output items.
        d_num_selected_out: Device array to store the number of items that passed the selection.
            The count is stored in ``d_num_selected_out[0]``.
        cond: Callable representing the selection condition (predicate). Should return a
            boolean-like value (typically uint8) where non-zero means the item passes the selection.
            Can reference device arrays as globals/closures - they will be automatically captured.
        num_items: Number of items in the input sequence.
        stream: CUDA stream to use for the operation (optional).
    """
    # Create adapter to support stateful ops
    cond_adapter = make_op_adapter(cond)
    selector = make_select(d_in, d_out, d_num_selected_out, cond_adapter)

    tmp_storage_bytes = selector(
        None,
        d_in,
        d_out,
        d_num_selected_out,
        cond_adapter,
        num_items,
        stream,
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    selector(
        tmp_storage,
        d_in,
        d_out,
        d_num_selected_out,
        cond_adapter,
        num_items,
        stream,
    )
