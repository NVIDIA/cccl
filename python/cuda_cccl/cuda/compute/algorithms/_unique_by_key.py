# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
from .._utils.protocols import (
    get_data_pointer,
    validate_and_get_stream,
)
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..iterators._iterators import IteratorBase
from ..op import OpAdapter, OpKind, make_op_adapter
from ..typing import DeviceArrayLike


def _make_cache_key(
    d_in_keys: DeviceArrayLike | IteratorBase,
    d_in_items: DeviceArrayLike | IteratorBase,
    d_out_keys: DeviceArrayLike | IteratorBase,
    d_out_items: DeviceArrayLike | IteratorBase,
    d_out_num_selected: DeviceArrayLike,
    op: OpAdapter,
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

    return (
        d_in_keys_key,
        d_in_items_key,
        d_out_keys_key,
        d_out_items_key,
        d_out_num_selected_key,
        op.get_cache_key(),
    )


class _UniqueByKey:
    __slots__ = [
        "build_result",
        "d_in_keys_cccl",
        "d_in_items_cccl",
        "d_out_keys_cccl",
        "d_out_items_cccl",
        "d_out_num_selected_cccl",
        "op",
        "op_cccl",
    ]

    def __init__(
        self,
        d_in_keys: DeviceArrayLike | IteratorBase,
        d_in_items: DeviceArrayLike | IteratorBase,
        d_out_keys: DeviceArrayLike | IteratorBase,
        d_out_items: DeviceArrayLike | IteratorBase,
        d_out_num_selected: DeviceArrayLike,
        op: OpAdapter,
    ):
        self.d_in_keys_cccl = cccl.to_cccl_input_iter(d_in_keys)
        self.d_in_items_cccl = cccl.to_cccl_input_iter(d_in_items)
        self.d_out_keys_cccl = cccl.to_cccl_output_iter(d_out_keys)
        self.d_out_items_cccl = cccl.to_cccl_output_iter(d_out_items)
        self.d_out_num_selected_cccl = cccl.to_cccl_output_iter(d_out_num_selected)

        # Compile the op - unique_by_key expects bool return (comparison)
        value_type = cccl.get_value_type(d_in_keys)
        self.op_cccl = op.compile((value_type, value_type), numba.types.uint8)

        self.build_result = call_build(
            _bindings.DeviceUniqueByKeyBuildResult,
            self.d_in_keys_cccl,
            self.d_in_items_cccl,
            self.d_out_keys_cccl,
            self.d_out_items_cccl,
            self.d_out_num_selected_cccl,
            self.op_cccl,
        )

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
        set_cccl_iterator_state(self.d_in_keys_cccl, d_in_keys)
        set_cccl_iterator_state(self.d_in_items_cccl, d_in_items)
        set_cccl_iterator_state(self.d_out_keys_cccl, d_out_keys)
        set_cccl_iterator_state(self.d_out_items_cccl, d_out_items)
        set_cccl_iterator_state(self.d_out_num_selected_cccl, d_out_num_selected)

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
            self.d_out_num_selected_cccl,
            self.op_cccl,
            num_items,
            stream_handle,
        )
        return temp_storage_bytes


@cache_with_key(_make_cache_key)
def _make_unique_by_key_cached(
    d_in_keys: DeviceArrayLike | IteratorBase,
    d_in_items: DeviceArrayLike | IteratorBase,
    d_out_keys: DeviceArrayLike | IteratorBase,
    d_out_items: DeviceArrayLike | IteratorBase,
    d_out_num_selected: DeviceArrayLike,
    op: OpAdapter,
):
    """Internal cached factory for _UniqueByKey."""
    return _UniqueByKey(
        d_in_keys, d_in_items, d_out_keys, d_out_items, d_out_num_selected, op
    )


def make_unique_by_key(
    d_in_keys: DeviceArrayLike | IteratorBase,
    d_in_items: DeviceArrayLike | IteratorBase,
    d_out_keys: DeviceArrayLike | IteratorBase,
    d_out_items: DeviceArrayLike | IteratorBase,
    d_out_num_selected: DeviceArrayLike,
    op: Callable | OpKind,
):
    """Implements a device-wide unique by key operation using ``d_in_keys`` and the comparison operator ``op``. Only the first key and its value from each run is selected and the total number of items selected is also reported.

    Example:
        Below, ``make_unique_by_key`` is used to create a unique by key object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/unique/unique_by_key_object.py
          :language: python
          :start-after: # example-begin


    Args:
        d_in_keys: Device array or iterator containing the input sequence of keys
        d_in_items: Device array or iterator that contains each key's corresponding item
        d_out_keys: Device array or iterator to store the outputted keys
        d_out_items: Device array or iterator to store each outputted key's item
        d_out_num_selected: Device array to store how many items were selected
        op: Callable or OpKind representing the equality operator

    Returns:
        A callable object that can be used to perform unique by key
    """
    op_adapter = make_op_adapter(op)
    return _make_unique_by_key_cached(
        d_in_keys,
        d_in_items,
        d_out_keys,
        d_out_items,
        d_out_num_selected,
        op_adapter,
    )


def unique_by_key(
    d_in_keys: DeviceArrayLike | IteratorBase,
    d_in_items: DeviceArrayLike | IteratorBase,
    d_out_keys: DeviceArrayLike | IteratorBase,
    d_out_items: DeviceArrayLike | IteratorBase,
    d_out_num_selected: DeviceArrayLike,
    op: Callable | OpKind,
    num_items: int,
    stream=None,
):
    """
    Performs device-wide unique by key operation using the single-phase API.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``unique_by_key`` is used to populate the arrays of output keys and items with the first key and its corresponding item from each sequence of equal keys. It also outputs the number of items selected.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/unique/unique_by_key_basic.py
            :language: python
            :start-after: # example-begin


    Args:
        d_in_keys: Device array or iterator containing the input sequence of keys
        d_in_items: Device array or iterator that contains each key's corresponding item
        d_out_keys: Device array or iterator to store the outputted keys
        d_out_items: Device array or iterator to store each outputted key's item
        d_out_num_selected: Device array to store how many items were selected
        op: Callable or OpKind representing the equality operator
        num_items: Number of items to process
        stream: CUDA stream for the operation (optional)
    """
    uniquer = make_unique_by_key(
        d_in_keys, d_in_items, d_out_keys, d_out_items, d_out_num_selected, op
    )
    tmp_storage_bytes = uniquer(
        None,
        d_in_keys,
        d_in_items,
        d_out_keys,
        d_out_items,
        d_out_num_selected,
        num_items,
        stream,
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    uniquer(
        tmp_storage,
        d_in_keys,
        d_in_items,
        d_out_keys,
        d_out_items,
        d_out_num_selected,
        num_items,
        stream,
    )
