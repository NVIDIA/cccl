# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import ClassVar

import numpy as np

from .. import _bindings, types
from .. import _cccl_interop as cccl
from .._caching import cache_build_results, cache_with_registered_key_functions
from .._cccl_interop import set_cccl_iterator_state
from .._serialization import (
    BUILD_RESULTS,
    ITER,
    OP,
    Serializable,
)
from .._utils import protocols
from ..op import OpAdapter, OpKind, make_op_adapter
from ..typing import DeviceArrayLike, IteratorT, Operator


def _data_pointer_or_none(array) -> int | None:
    # A ProxyArray is a build-time placeholder with no GPU allocation and thus no
    # data pointer; return None for it (these pointers are only cache-key
    # discriminators) so binary_search can be built without a GPU.
    from .._proxy import is_proxy

    return None if is_proxy(array) else protocols.get_data_pointer(array)


class _BinarySearch:
    # Shared implementation for the lower/upper bound searchers.
    _MODE: ClassVar[_bindings.BinarySearchMode]

    __slots__ = [
        "_bound_build_result",
        "build_results",
        "loaded_build_result",
        "d_data_cccl",
        "d_values_cccl",
        "d_out_cccl",
        "op_cccl",
        "data_ptr",
        "out_ptr",
    ]

    __serialization_schema__ = (
        ("d_data_cccl", ITER),
        ("d_values_cccl", ITER),
        ("d_out_cccl", ITER),
        ("op_cccl", OP),
        ("build_results", BUILD_RESULTS(_bindings.DeviceBinarySearchBuildResult)),
    )

    def __init__(
        self,
        d_data: DeviceArrayLike,
        d_values: DeviceArrayLike | IteratorT,
        d_out: DeviceArrayLike,
        comp: OpAdapter,
        compute_capability=None,
    ):
        if not protocols.is_device_array(d_data):
            raise ValueError("d_data must be a device array for index outputs.")
        if not protocols.is_device_array(d_out):
            raise ValueError("d_out must be a device array for index outputs.")

        out_dtype = protocols.get_dtype(d_out)
        if out_dtype.kind != "u":
            raise TypeError("d_out must use an unsigned integer dtype for indices.")
        if out_dtype.itemsize != np.dtype(np.uintp).itemsize:
            raise ValueError(
                "d_out must use a pointer-sized unsigned integer dtype (np.uintp)."
            )

        self.data_ptr = _data_pointer_or_none(d_data)
        self.out_ptr = _data_pointer_or_none(d_out)

        self.d_data_cccl = cccl.to_cccl_input_iter(d_data)
        self.d_values_cccl = cccl.to_cccl_input_iter(d_values)
        data_value_type = cccl.get_value_type(d_data)
        self.d_out_cccl = cccl.to_cccl_output_iter(d_out)

        self.op_cccl = comp.compile((data_value_type, data_value_type), types.uint8)

        self.build_results, self._bound_build_result = cache_build_results(
            _bindings.DeviceBinarySearchBuildResult,
            d_data,
            d_values,
            d_out,
            comp,
            self._MODE,
            compute_capability=compute_capability,
            builder=lambda: cccl.build_for_ccs(
                _bindings.DeviceBinarySearchBuildResult,
                self._MODE,
                self.d_data_cccl,
                self.d_values_cccl,
                self.d_out_cccl,
                self.op_cccl,
                compute_capability=compute_capability,
            ),
        )

    def __call__(
        self,
        *,
        d_data,
        num_items: int,
        d_values,
        num_values: int,
        d_out,
        comp: Operator | None,
        stream=None,
    ):
        # Select (and lazily load) the build result for the current device.
        self.loaded_build_result = cccl.resolve_build_result(
            self.build_results, self._bound_build_result
        )

        set_cccl_iterator_state(self.d_data_cccl, d_data)
        set_cccl_iterator_state(self.d_values_cccl, d_values)
        set_cccl_iterator_state(self.d_out_cccl, d_out)

        # Update op state for stateful ops
        comp_adapter = make_op_adapter(OpKind.LESS if comp is None else comp)
        self.op_cccl.state = comp_adapter.get_state()

        stream_handle = protocols.validate_and_get_stream(stream)
        self.loaded_build_result.compute(
            self.d_data_cccl,
            num_items,
            self.d_values_cccl,
            num_values,
            self.d_out_cccl,
            self.op_cccl,
            stream_handle,
        )


class _LowerBound(_BinarySearch, Serializable):
    __slots__ = ()
    _MODE = _bindings.BinarySearchMode.LOWER_BOUND


class _UpperBound(_BinarySearch, Serializable):
    __slots__ = ()
    _MODE = _bindings.BinarySearchMode.UPPER_BOUND


@cache_with_registered_key_functions
def _make_binary_search(
    d_data: DeviceArrayLike,
    d_values: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike,
    comp: OpAdapter,
    mode: _bindings.BinarySearchMode,
    data_ptr: int,
    out_ptr: int,
    compute_capability=None,
):
    """Cached factory for the binary_search searchers."""
    cls = _LowerBound if mode == _bindings.BinarySearchMode.LOWER_BOUND else _UpperBound
    return cls(d_data, d_values, d_out, comp, compute_capability=compute_capability)


def make_lower_bound(
    *,
    d_data: DeviceArrayLike,
    d_values: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike,
    comp: Operator | None = None,
    compute_capability=None,
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
        compute_capability: Compute capability, or list of capabilities, to
            build for ahead of time. Accepts a packed int (e.g. ``90``), a
            ``(major, minor)`` pair, a string (e.g. ``"9.0"``), or a list
            thereof. When ``None`` (the default), the current device's
            architecture is used.

    Returns:
        A callable object that performs lower_bound.

    See Also:
        :func:`lower_bound`
    """
    comp_adapter = make_op_adapter(OpKind.LESS if comp is None else comp)
    return _make_binary_search(
        d_data,
        d_values,
        d_out,
        comp_adapter,
        _bindings.BinarySearchMode.LOWER_BOUND,
        _data_pointer_or_none(d_data),
        _data_pointer_or_none(d_out),
        compute_capability=compute_capability,
    )


def make_upper_bound(
    *,
    d_data: DeviceArrayLike,
    d_values: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike,
    comp: Operator | None = None,
    compute_capability=None,
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
        compute_capability: Compute capability, or list of capabilities, to
            build for ahead of time. Accepts a packed int (e.g. ``90``), a
            ``(major, minor)`` pair, a string (e.g. ``"9.0"``), or a list
            thereof. When ``None`` (the default), the current device's
            architecture is used.

    Returns:
        A callable object that performs upper_bound.

    See Also:
        :func:`upper_bound`
    """
    comp_adapter = make_op_adapter(OpKind.LESS if comp is None else comp)
    return _make_binary_search(
        d_data,
        d_values,
        d_out,
        comp_adapter,
        _bindings.BinarySearchMode.UPPER_BOUND,
        _data_pointer_or_none(d_data),
        _data_pointer_or_none(d_out),
        compute_capability=compute_capability,
    )


def lower_bound(
    *,
    d_data: DeviceArrayLike,
    num_items: int,
    d_values: DeviceArrayLike | IteratorT,
    num_values: int,
    d_out: DeviceArrayLike,
    comp: Operator | None = None,
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
        num_items: Number of items in ``d_data``.
        d_values: Device array or iterator containing the search values.
        num_values: Number of items in ``d_values``.
        d_out: Device array to store the index results.
        comp: Optional comparison operator (default: ``OpKind.LESS``).
        stream: CUDA stream for the operation (optional).
    """
    searcher = make_lower_bound(
        d_data=d_data, d_values=d_values, d_out=d_out, comp=comp
    )
    searcher(
        d_data=d_data,
        num_items=num_items,
        d_values=d_values,
        num_values=num_values,
        d_out=d_out,
        comp=comp,
        stream=stream,
    )


def upper_bound(
    *,
    d_data: DeviceArrayLike,
    num_items: int,
    d_values: DeviceArrayLike | IteratorT,
    num_values: int,
    d_out: DeviceArrayLike,
    comp: Operator | None = None,
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
        num_items: Number of items in ``d_data``.
        d_values: Device array or iterator containing the search values.
        num_values: Number of items in ``d_values``.
        d_out: Device array to store the index results.
        comp: Optional comparison operator (default: ``OpKind.LESS``).
        stream: CUDA stream for the operation (optional).
    """
    searcher = make_upper_bound(
        d_data=d_data, d_values=d_values, d_out=d_out, comp=comp
    )
    searcher(
        d_data=d_data,
        num_items=num_items,
        d_values=d_values,
        num_values=num_values,
        d_out=d_out,
        comp=comp,
        stream=stream,
    )
