# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete block Run Length Decode operations for CuTe kernels."""

from cuda.coop._core.thread_group import ThreadGroup

from ._temp_storage import TempStorage
from ._thread_data import _snapshot_readable_payload


def _decode(
    group,
    run_values,
    run_lengths,
    *,
    decoded_items_per_thread,
    offset,
    destination,
    bulk,
    temp_storage,
):
    if not isinstance(group, ThreadGroup):
        raise TypeError("run_length_decode group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            "run_length_decode requires a complete this_block() group"
        )
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError("run_length_decode temp_storage must be CUTLASS TempStorage")
    primitive = "run_length_decode_into" if bulk else "run_length_decode"
    values = _snapshot_readable_payload(
        run_values, name="run_values", primitive=primitive
    )
    lengths = _snapshot_readable_payload(
        run_lengths, name="run_lengths", primitive=primitive
    )
    if values.items_per_thread != lengths.items_per_thread:
        raise ValueError(
            "run_length_decode run inputs must have matching items_per_thread"
        )

    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._run_length import provider_run_length_decode

    return provider_run_length_decode(
        group=group,
        launch=current_kernel_launch_facts(),
        values=values,
        lengths=lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        offset=offset,
        destination=destination,
        bulk=bulk,
        temp_storage=temp_storage,
    )


def run_length_decode(
    group,
    run_values,
    run_lengths,
    /,
    *,
    decoded_items_per_thread,
    decoded_window_offset=0,
    temp_storage=None,
):
    """Decode a blocked window without changing either input payload.

    Participation, length validation, window offsets, zero-filled tails, and
    scratch reuse follow :func:`cuda.coop.run_length_decode`. The qualified
    overload also accepts CuTe register-memory tensors and ``TensorSSA``
    values through :meth:`cuda.coop.cutlass.ThreadData.from_payload`.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block. Every thread must participate.
    run_values, run_lengths : ThreadData or CuTe register payload
        Equal fixed-size per-thread payloads in blocked order. Values use the
        common numeric dtypes; lengths use signed or unsigned integers up to
        64 bits. Positive lengths precede all trailing zero padding.
    decoded_items_per_thread : int
        Positive compile-time output extent, independent of the run extent.
    decoded_window_offset : integer, optional
        Block-uniform nonnegative start in the decoded stream, default zero.
        Static and runtime integers up to uint64 retain their width through
        validation. A window starting beyond the stream contains zeros.
    temp_storage : TempStorage, optional
        Explicit block scratch, or omit for automatic allocation. Requested
        alignment is a minimum. Synchronize before reuse with ``auto_sync=False``.

    Returns
    -------
    ThreadData
        Fresh payload with the run-value dtype and requested output extent.
        Neither input is modified. The decoded total must fit uint32; negative
        lengths, misplaced zero padding, and overflow trap before decoding.

    Notes
    -----
    See the :doc:`Run Length Decode visualization
    <coop/visualizations/run-length-decode>` for windows and zero-filled tails.
    """
    return _decode(
        group,
        run_values,
        run_lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        offset=decoded_window_offset,
        destination=None,
        bulk=False,
        temp_storage=temp_storage,
    )


def run_length_decode_into(
    group,
    run_values,
    run_lengths,
    destination,
    /,
    *,
    decoded_items_per_thread,
    destination_offset=0,
    temp_storage=None,
):
    """Decode a full stream into a CuTe global-memory tensor.

    Length validation, input preservation, offset/capacity checks, and scratch
    lifetime follow :func:`cuda.coop.run_length_decode_into`. The prepared CUB
    table remains live through all internal decode windows.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block. Every thread must participate.
    run_values, run_lengths : ThreadData or CuTe register payload
        Matching blocked run payloads, as in
        :func:`cuda.coop.cutlass.run_length_decode`.
    destination : cute.Tensor
        Contiguous one-dimensional global-memory tensor with the run-value
        dtype and a known static or runtime extent. All block members supply
        the same tensor. Bare pointers lack the capacity needed for validation.
    decoded_items_per_thread : int
        Positive compile-time number of items per thread in each decode window.
    destination_offset : integer, optional
        Block-uniform nonnegative output index, default zero. Static and
        runtime integers up to uint64 are checked before pointer arithmetic.
    temp_storage : TempStorage, optional
        Scratch covering the complete operation. Synchronize before reuse when
        its descriptor sets ``auto_sync=False``.

    Returns
    -------
    cutlass.Uint32
        Full decoded size, available to every member. Empty input writes
        nothing. Insufficient capacity traps before any output write; elements
        outside the decoded interval remain unchanged. The total must fit uint32.

    Notes
    -----
    See the :doc:`Run Length Decode visualization
    <coop/visualizations/run-length-decode>` for bulk output and run ordering.
    """
    return _decode(
        group,
        run_values,
        run_lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        offset=destination_offset,
        destination=destination,
        bulk=True,
        temp_storage=temp_storage,
    )


__all__ = ["run_length_decode", "run_length_decode_into"]
