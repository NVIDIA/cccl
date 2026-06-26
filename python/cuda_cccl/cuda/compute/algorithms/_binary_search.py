# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import numpy as np

from .. import _bindings, types
from .. import _cccl_interop as cccl
from .._caching import cache_with_registered_key_functions
from .._cccl_interop import call_build, set_cccl_iterator_state
from .._utils import protocols
from ..op import OpAdapter, OpKind, make_op_adapter
from ..typing import DeviceArrayLike, IteratorT, Operator

# 4-byte mode tags prepended by serialize() so load_lower_bound / load_upper_bound
# can reject a blob that was saved for the opposite mode.
_LOWER_BOUND_TAG = b"LBND"
_UPPER_BOUND_TAG = b"UBND"
_MODE_TAGS = {
    _bindings.BinarySearchMode.LOWER_BOUND: _LOWER_BOUND_TAG,
    _bindings.BinarySearchMode.UPPER_BOUND: _UPPER_BOUND_TAG,
}
_TAG_MODES = {v: k for k, v in _MODE_TAGS.items()}


class _BinarySearch:
    __slots__ = [
        "build_result",
        "d_data_cccl",
        "d_values_cccl",
        "d_out_cccl",
        "op_cccl",
        "data_ptr",
        "out_ptr",
        "mode",
    ]

    def __init__(
        self,
        d_data: DeviceArrayLike,
        d_values: DeviceArrayLike | IteratorT,
        d_out: DeviceArrayLike,
        comp: OpAdapter,
        mode: _bindings.BinarySearchMode,
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

        self.data_ptr = protocols.get_data_pointer(d_data)
        self.out_ptr = protocols.get_data_pointer(d_out)
        self.mode = mode

        self.d_data_cccl = cccl.to_cccl_input_iter(d_data)
        self.d_values_cccl = cccl.to_cccl_input_iter(d_values)
        data_value_type = cccl.get_value_type(d_data)
        self.d_out_cccl = cccl.to_cccl_output_iter(d_out)

        self.op_cccl = comp.compile((data_value_type, data_value_type), types.uint8)

        self.build_result = call_build(
            _bindings.DeviceBinarySearchBuildResult,
            mode,
            self.d_data_cccl,
            self.d_values_cccl,
            self.d_out_cccl,
            self.op_cccl,
        )

    @classmethod
    def _deserialize(
        cls,
        blob: bytes,
        d_data: DeviceArrayLike,
        d_values: DeviceArrayLike | IteratorT,
        d_out: DeviceArrayLike,
        comp: Operator | None = None,
        expected_mode: "_bindings.BinarySearchMode | None" = None,
    ) -> "_BinarySearch":
        """Reconstruct a binary_search from a blob produced by :meth:`serialize`."""
        if len(blob) < 4:
            raise ValueError("AoT blob is too short to contain a mode tag.")
        tag = blob[:4]
        mode = _TAG_MODES.get(tag)
        if mode is None:
            raise ValueError(
                f"AoT blob has unrecognized mode tag {tag!r}; "
                "was it produced by binary_search.serialize()?"
            )
        if expected_mode is not None and mode != expected_mode:
            actual = (
                "lower_bound"
                if mode == _bindings.BinarySearchMode.LOWER_BOUND
                else "upper_bound"
            )
            wanted = (
                "lower_bound"
                if expected_mode == _bindings.BinarySearchMode.LOWER_BOUND
                else "upper_bound"
            )
            raise ValueError(
                f"AoT blob mode mismatch: blob was saved as {actual!r}, "
                f"but loaded through {wanted!r}."
            )
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
        obj = cls.__new__(cls)
        obj.data_ptr = protocols.get_data_pointer(d_data)
        obj.out_ptr = protocols.get_data_pointer(d_out)
        obj.mode = mode
        obj.d_data_cccl = cccl.to_cccl_input_iter(d_data)
        obj.d_values_cccl = cccl.to_cccl_input_iter(d_values)
        data_value_type = cccl.get_value_type(d_data)
        obj.d_out_cccl = cccl.to_cccl_output_iter(d_out)
        comp_adapter = make_op_adapter(OpKind.LESS if comp is None else comp)
        obj.op_cccl = comp_adapter.compile_for_load(
            (data_value_type, data_value_type), types.uint8
        )
        obj.build_result = _bindings.DeviceBinarySearchBuildResult.deserialize(blob[4:])
        return obj

    def serialize(self) -> bytes:
        """Return a bytes blob representing this built binary_search.

        The blob encodes the search mode (lower_bound or upper_bound).
        Use :func:`load_lower_bound` or :func:`load_upper_bound` to reconstruct.
        """
        return _MODE_TAGS[self.mode] + self.build_result.serialize()

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
        set_cccl_iterator_state(self.d_data_cccl, d_data)
        set_cccl_iterator_state(self.d_values_cccl, d_values)
        set_cccl_iterator_state(self.d_out_cccl, d_out)

        # Update op state for stateful ops
        comp_adapter = make_op_adapter(OpKind.LESS if comp is None else comp)
        self.op_cccl.state = comp_adapter.get_state()

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


@cache_with_registered_key_functions
def _make_binary_search(
    d_data: DeviceArrayLike,
    d_values: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike,
    comp: OpAdapter,
    mode: _bindings.BinarySearchMode,
    data_ptr: int,
    out_ptr: int,
):
    """Cached factory for _BinarySearch."""
    return _BinarySearch(d_data, d_values, d_out, comp, mode)


def make_lower_bound(
    *,
    d_data: DeviceArrayLike,
    d_values: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike,
    comp: Operator | None = None,
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
    comp_adapter = make_op_adapter(OpKind.LESS if comp is None else comp)
    return _make_binary_search(
        d_data,
        d_values,
        d_out,
        comp_adapter,
        _bindings.BinarySearchMode.LOWER_BOUND,
        protocols.get_data_pointer(d_data),
        protocols.get_data_pointer(d_out),
    )


def make_upper_bound(
    *,
    d_data: DeviceArrayLike,
    d_values: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike,
    comp: Operator | None = None,
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
    comp_adapter = make_op_adapter(OpKind.LESS if comp is None else comp)
    return _make_binary_search(
        d_data,
        d_values,
        d_out,
        comp_adapter,
        _bindings.BinarySearchMode.UPPER_BOUND,
        protocols.get_data_pointer(d_data),
        protocols.get_data_pointer(d_out),
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


def load_lower_bound(
    blob: bytes,
    *,
    d_data: DeviceArrayLike,
    d_values: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike,
    comp: Operator | None = None,
):
    """Reconstruct a lower_bound searcher from a blob produced by ``searcher.serialize()``.

    Raises ``ValueError`` if the blob was produced by an upper_bound searcher.

    Args:
        blob: Bytes blob produced by a lower_bound searcher's ``serialize()`` method.
        d_data: Device array containing the sorted input range.
        d_values: Device array or iterator containing the search values.
        d_out: Device array to store the index results.
        comp: Optional comparison operator (default: ``OpKind.LESS``).

    Returns:
        A callable lower_bound searcher.
    """
    return _BinarySearch._deserialize(
        blob,
        d_data,
        d_values,
        d_out,
        comp,
        expected_mode=_bindings.BinarySearchMode.LOWER_BOUND,
    )


def load_upper_bound(
    blob: bytes,
    *,
    d_data: DeviceArrayLike,
    d_values: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike,
    comp: Operator | None = None,
):
    """Reconstruct an upper_bound searcher from a blob produced by ``searcher.serialize()``.

    Raises ``ValueError`` if the blob was produced by a lower_bound searcher.

    Args:
        blob: Bytes blob produced by an upper_bound searcher's ``serialize()`` method.
        d_data: Device array containing the sorted input range.
        d_values: Device array or iterator containing the search values.
        d_out: Device array to store the index results.
        comp: Optional comparison operator (default: ``OpKind.LESS``).

    Returns:
        A callable upper_bound searcher.
    """
    return _BinarySearch._deserialize(
        blob,
        d_data,
        d_values,
        d_out,
        comp,
        expected_mode=_bindings.BinarySearchMode.UPPER_BOUND,
    )
