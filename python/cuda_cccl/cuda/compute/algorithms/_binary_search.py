# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Callable

import numba
import numpy as np

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import cache_with_key
from .._cccl_interop import call_build, set_cccl_iterator_state
from .._utils import protocols
from ..iterators._iterators import IteratorBase
from ..op import OpAdapter, OpKind, make_op_adapter
from ..typing import DeviceArrayLike


def _make_cache_key(
    d_data: DeviceArrayLike | IteratorBase,
    d_values: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    comp: OpAdapter,
    mode: _bindings.BinarySearchMode,
):
    d_data_key = (
        d_data.kind if isinstance(d_data, IteratorBase) else protocols.get_dtype(d_data)
    )
    d_values_key = (
        d_values.kind
        if isinstance(d_values, IteratorBase)
        else protocols.get_dtype(d_values)
    )
    d_out_key = (
        d_out.kind if isinstance(d_out, IteratorBase) else protocols.get_dtype(d_out)
    )
    data_ptr = (
        None if isinstance(d_data, IteratorBase) else protocols.get_data_pointer(d_data)
    )
    out_ptr = (
        None if isinstance(d_out, IteratorBase) else protocols.get_data_pointer(d_out)
    )

    return (
        d_data_key,
        d_values_key,
        d_out_key,
        data_ptr,
        out_ptr,
        comp.get_cache_key(),
        mode,
    )


def _normalize_comp(comp: Callable | OpKind | None) -> OpAdapter:
    # Use a lambda for the default comparator rather than OpKind.LESS
    # because well-known ops don't carry type information needed by
    # the binary search JIT compilation.
    if comp is None or comp is OpKind.LESS:

        def _default_less(a, b):
            return a < b

        return make_op_adapter(_default_less)
    return make_op_adapter(comp)


class _BinarySearch:
    __slots__ = [
        "build_result",
        "d_data_cccl",
        "d_values_cccl",
        "d_out_cccl",
        "op_cccl",
        "data_ptr",
        "out_ptr",
    ]

    def __init__(
        self,
        d_data: DeviceArrayLike,
        d_values: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike,
        comp: OpAdapter,
        mode: _bindings.BinarySearchMode,
    ):
        if isinstance(d_data, IteratorBase):
            raise ValueError("d_data must be a device array for index outputs.")
        if isinstance(d_out, IteratorBase):
            raise ValueError("d_out must be a device array for index outputs.")

        out_dtype = protocols.get_dtype(d_out)
        if out_dtype.kind != "u":
            raise TypeError("d_out must use an unsigned integer dtype for indices.")
        if out_dtype.itemsize != np.dtype(np.uintp).itemsize:
            raise ValueError(
                "d_out must use a pointer-sized unsigned integer dtype (np.uintp)."
            )

        self.data_ptr = protocols.get_data_pointer(d_data)
        self.out_ptr = protocols.get_data_pointer(d_out)

        self.d_data_cccl = cccl.to_cccl_input_iter(d_data)
        self.d_values_cccl = cccl.to_cccl_input_iter(d_values)
        data_value_type = cccl.get_value_type(d_data)
        self.d_out_cccl = cccl.to_cccl_output_iter(d_out)

        self.op_cccl = comp.compile(
            (data_value_type, data_value_type), numba.types.uint8
        )

        self.build_result = call_build(
            _bindings.DeviceBinarySearchBuildResult,
            mode,
            self.d_data_cccl,
            self.d_values_cccl,
            self.d_out_cccl,
            self.op_cccl,
        )

    def __call__(
        self,
        d_data,
        d_values,
        d_out,
        num_items: int,
        num_values: int,
        stream=None,
    ):
        if protocols.get_data_pointer(d_data) != self.data_ptr:
            raise ValueError(
                "d_data must match the array used to construct the binary search object."
            )
        if protocols.get_data_pointer(d_out) != self.out_ptr:
            raise ValueError(
                "d_out must match the array used to construct the binary search object."
            )

        set_cccl_iterator_state(self.d_data_cccl, d_data)
        set_cccl_iterator_state(self.d_values_cccl, d_values)
        set_cccl_iterator_state(self.d_out_cccl, d_out)

        stream_handle = protocols.validate_and_get_stream(stream)
        self.build_result.compute(
            self.d_data_cccl,
            num_items,
            self.d_values_cccl,
            num_values,
            self.d_out_cccl,
            self.op_cccl,
            stream_handle,
        )


@cache_with_key(_make_cache_key)
def _make_binary_search_cached(
    d_data: DeviceArrayLike,
    d_values: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    comp: OpAdapter,
    mode: _bindings.BinarySearchMode,
):
    """Internal cached factory for _BinarySearch."""
    return _BinarySearch(d_data, d_values, d_out, comp, mode)


def make_lower_bound(
    d_data: DeviceArrayLike,
    d_values: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    comp: Callable | OpKind | None = None,
):
    """
    Create a lower_bound object that can be called to find insertion positions.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/binary_search/lower_bound_object.py
            :language: python
            :start-after: # example-begin

    Args:
        d_data: Device array containing the sorted input range.
        d_values: Device array or iterator containing the search values.
        d_out: Device array to store the index results.
        comp: Optional comparison operator (default: ``OpKind.LESS``).

    Returns:
        A callable object that performs lower_bound.

    See Also:
        :func:`lower_bound`
    """
    comp_adapter = _normalize_comp(comp)
    return _make_binary_search_cached(
        d_data,
        d_values,
        d_out,
        comp_adapter,
        _bindings.BinarySearchMode.LOWER_BOUND_INDEX,
    )


def make_upper_bound(
    d_data: DeviceArrayLike,
    d_values: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    comp: Callable | OpKind | None = None,
):
    """
    Create an upper_bound object that can be called to find insertion positions.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/binary_search/upper_bound_object.py
            :language: python
            :start-after: # example-begin

    Args:
        d_data: Device array containing the sorted input range.
        d_values: Device array or iterator containing the search values.
        d_out: Device array to store the index results.
        comp: Optional comparison operator (default: ``OpKind.LESS``).

    Returns:
        A callable object that performs upper_bound.

    See Also:
        :func:`upper_bound`
    """
    comp_adapter = _normalize_comp(comp)
    return _make_binary_search_cached(
        d_data,
        d_values,
        d_out,
        comp_adapter,
        _bindings.BinarySearchMode.UPPER_BOUND_INDEX,
    )


def lower_bound(
    d_data: DeviceArrayLike,
    d_values: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    num_items: int,
    num_values: int,
    comp: Callable | OpKind | None = None,
    stream=None,
):
    """
    Find the *first* position that each value in ``d_values`` would be inserted into
    ``d_data`` to maintain sorted order.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/binary_search/lower_bound_basic.py
            :language: python
            :start-after: # example-begin

    Args:
        d_data: Device array containing the sorted input range.
        d_values: Device array or iterator containing the search values.
        d_out: Device array to store the index results.
        num_items: Number of items in ``d_data``.
        num_values: Number of items in ``d_values``.
        comp: Optional comparison operator (default: ``OpKind.LESS``).
        stream: CUDA stream for the operation (optional).
    """
    searcher = make_lower_bound(d_data, d_values, d_out, comp)
    searcher(d_data, d_values, d_out, num_items, num_values, stream)


def upper_bound(
    d_data: DeviceArrayLike,
    d_values: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    num_items: int,
    num_values: int,
    comp: Callable | OpKind | None = None,
    stream=None,
):
    """
    Find the *last* position that each value in ``d_values`` would be inserted into
    ``d_data`` to maintain sorted order.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/binary_search/upper_bound_basic.py
            :language: python
            :start-after: # example-begin

    Args:
        d_data: Device array containing the sorted input range.
        d_values: Device array or iterator containing the search values.
        d_out: Device array to store the index results.
        num_items: Number of items in ``d_data``.
        num_values: Number of items in ``d_values``.
        comp: Optional comparison operator (default: ``OpKind.LESS``).
        stream: CUDA stream for the operation (optional).
    """
    searcher = make_upper_bound(d_data, d_values, d_out, comp)
    searcher(d_data, d_values, d_out, num_items, num_values, stream)
