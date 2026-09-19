# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block Run Length Decode operations."""

from __future__ import annotations

from typing import Any

from ..dtype_policy import validate_portable_integer_value_dtype_name
from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _group_primitive_marker,
    _portable_group_operation,
)
from ._payload import _validate_common_numeric_value


@_portable_group_operation("run_length_decode", group_kinds=("block",))
def run_length_decode(
    group: ThreadGroup,
    run_values: Any,
    run_lengths: Any,
    /,
    *,
    decoded_items_per_thread: int,
    decoded_window_offset: Any = 0,
    temp_storage: Any = None,
) -> Any:
    """Return a fresh blocked window of the decoded run stream.

    Parameters
    ----------
    group : ThreadGroup
        A complete one-dimensional ``this_block()`` group. Every member must
        participate in the same call.
    run_values : ThreadDataLike
        Readable fixed-size per-thread run values in blocked order. Thread
        ``t`` owns runs ``t * runs_per_thread + i``. Values may use signed or
        unsigned 8-, 16-, 32-, or 64-bit integers, float32, or float64.
    run_lengths : ThreadDataLike
        Readable integer lengths with the same per-thread extent as
        ``run_values``. Signed and unsigned integer dtypes up to 64 bits are
        supported. Across the block, positive lengths must precede all zero
        lengths; zeros pad the end of the run tile. An all-zero tile is empty.
    decoded_items_per_thread : int
        Positive compile-time extent of each returned per-thread payload.
        It is independent of the number of input runs per thread.
    decoded_window_offset : integer, optional
        Block-uniform nonnegative index of the first decoded item in the
        window, default zero. Thread ``t`` receives items beginning at
        ``decoded_window_offset + t * decoded_items_per_thread``. Offsets at
        or beyond the total decoded size produce an all-zero window.
        Integer controls retain their width through validation, up to uint64.
    temp_storage : TempStorageLike, optional
        Explicit block scratch descriptor. Omit it to allocate scratch
        automatically. With ``auto_sync=False``, synchronize the block before
        reusing that descriptor in a later collective.

    Returns
    -------
    ThreadDataLike
        A new payload with the value dtype and ``decoded_items_per_thread``
        items per member. Slots beyond the decoded stream contain zero. Both
        input payloads are preserved.

    Notes
    -----
    The decoded total must fit uint32. Negative lengths, a positive length
    after zero padding, and overflow trap before CUB is called. Invalid
    static controls are rejected during compilation; negative runtime offsets
    trap before writing outputs. Both run and window tile extents must fit
    signed 32-bit integers.

    Each call prepares its own CUB run table. Use
    :func:`cuda.coop.run_length_decode_into` to write a full stream while preparing
    that table once, or the qualified operation for total-size and relative
    run-offset outputs.
    """
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "run_length_decode",
            "run_values",
            run_values,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
        length_dtype = _validate_common_numeric_value(
            "run_length_decode",
            "run_lengths",
            run_lengths,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
        validate_portable_integer_value_dtype_name(
            length_dtype, operation="run_length_decode", parameter="run_lengths"
        )
    return _group_primitive_marker(
        "run_length_decode",
        group,
        run_values,
        run_lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        decoded_window_offset=decoded_window_offset,
        temp_storage=temp_storage,
    )


@_portable_group_operation("run_length_decode_into", group_kinds=("block",))
def run_length_decode_into(
    group: ThreadGroup,
    run_values: Any,
    run_lengths: Any,
    destination: Any,
    /,
    *,
    decoded_items_per_thread: int,
    destination_offset: Any = 0,
    temp_storage: Any = None,
) -> Any:
    """Decode a complete run stream into an array and return its total size.

    Parameters
    ----------
    group : ThreadGroup
        A complete one-dimensional ``this_block()`` group. Every member must
        participate in the same call.
    run_values, run_lengths : ThreadDataLike
        Matching readable per-thread run payloads in blocked order. Dtypes,
        positive-prefix lengths, trailing zero padding, empty input, and
        total-size limits follow :func:`cuda.coop.run_length_decode`.
    destination : array
        Writable contiguous one-dimensional global array with the run-value
        dtype. All block members must supply the same destination. Its
        remaining capacity after ``destination_offset`` must hold the entire
        decoded stream. It must not overlap either run input.
    decoded_items_per_thread : int
        Positive compile-time number of items per member in each internal
        window. The provider prepares the CUB run table once, then decodes
        windows of ``block_threads * decoded_items_per_thread`` items.
    destination_offset : integer, optional
        Block-uniform nonnegative destination index for the first decoded
        item, default zero. This is an output offset, not an input window
        offset. The full stream is decoded. Integer controls up to uint64 are
        checked before narrowing or pointer arithmetic.
    temp_storage : TempStorageLike, optional
        Explicit scratch for the entire operation. The prepared table remains
        live through all internal windows. The descriptor can be reused after
        the call; with ``auto_sync=False``, first synchronize the block.

    Returns
    -------
    uint32
        Total decoded size, with the same value available to every member.
        Empty input returns zero and leaves the destination unchanged.

    Notes
    -----
    Capacity and offset checks complete before any output write. Invalid runtime
    controls or insufficient capacity trap. Only the interval beginning at
    ``destination_offset`` and containing the returned number of items is
    written; the remaining destination elements are preserved. The last
    internal window is masked when the stream is not a whole number of
    windows. Both run inputs are preserved.
    """
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "run_length_decode_into",
            "run_values",
            run_values,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
        length_dtype = _validate_common_numeric_value(
            "run_length_decode_into",
            "run_lengths",
            run_lengths,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
        validate_portable_integer_value_dtype_name(
            length_dtype, operation="run_length_decode", parameter="run_lengths"
        )
    return _group_primitive_marker(
        "run_length_decode_into",
        group,
        run_values,
        run_lengths,
        destination,
        decoded_items_per_thread=decoded_items_per_thread,
        destination_offset=destination_offset,
        temp_storage=temp_storage,
    )


__all__ = ["run_length_decode", "run_length_decode_into"]
