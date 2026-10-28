# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS entry points for block run-length decoding.

Each call copies the per-thread run values and lengths from ThreadData, CuTe
register tensors, or TensorSSA values, so inputs stay unchanged. Values and
lengths must have the same per-thread count. run_length_decode returns one
decoded window as ThreadData. run_length_decode_into writes the whole stream
to a global tensor and returns its length.
"""

from __future__ import annotations

from typing import TypeVar

from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import CompilerIntegerLike

from ._temp_storage import TempStorage
from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
    _snapshot_readable_payload,
)
from ._typing import (
    BlockGroup,
    CommonNumericScalar,
    CommonThreadDataLike,
    IntegralScalar,
    TempStorageLike,
)

_LengthT = TypeVar("_LengthT", bound=IntegralScalar)

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)


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
    """Validate the block group and copy both run payloads.

    Values and lengths must have the same per-thread extent. The decoded
    extent is a separate compile-time control. The provider resolves the
    value and length types, the static or runtime offset type, and the
    destination checks. It then selects the window or bulk C++ wrapper.
    """

    primitive = "run_length_decode_into" if bulk else "run_length_decode"
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{primitive} group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            f"{primitive} requires a complete this_block() group"
        )
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError(f"{primitive} temp_storage must be CUTLASS TempStorage")
    values = _snapshot_readable_payload(
        run_values, name="run_values", primitive=primitive
    )
    lengths = _snapshot_readable_payload(
        run_lengths, name="run_lengths", primitive=primitive
    )
    if values.items_per_thread != lengths.items_per_thread:
        raise ValueError(
            f"{primitive} run inputs must have matching items_per_thread"
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
    group: BlockGroup,
    run_values: CommonThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    run_lengths: CommonThreadDataLike[_LengthT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    decoded_items_per_thread: int,
    decoded_window_offset: IntegralScalar = 0,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData:
    """Decode a blocked window without changing either input payload.

    CuTe register-memory tensors and ``TensorSSA`` values are adapted through
    :meth:`cuda.coop.cutlass.ThreadData.from_payload`. The window may start
    inside a run and may extend past the decoded stream.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block. Every thread must participate.
    run_values, run_lengths : ThreadData or CuTe register payload
        Equal fixed-size per-thread payloads in blocked order. Values use the
        signed or unsigned 8-, 16-, 32-, or 64-bit integers, ``Float32``, or
        ``Float64``. Lengths use signed or unsigned integers up to 64 bits.
        Positive lengths precede all trailing zero padding; an all-zero tile
        represents an empty stream.
    decoded_items_per_thread : int
        Positive compile-time output extent, independent of the run extent.
    decoded_window_offset : integer, optional
        Block-uniform nonnegative start in the decoded stream, default zero.
        Member ``t`` starts at this offset plus
        ``t * decoded_items_per_thread``. Static and runtime integers up to
        uint64 retain their width through validation. A window starting beyond
        the stream contains zeros.
    temp_storage : TempStorage, optional
        Explicit block scratch, or omit for automatic allocation. Requested
        alignment is a minimum. Synchronize before reuse with
        ``auto_sync=False``.

    Returns
    -------
    ThreadData
        Fresh payload with the run-value dtype and requested output extent.
        Slots beyond the decoded stream contain zero.
        Neither input is modified. The decoded total must fit uint32; negative
        lengths, misplaced zero padding, and overflow trap before decoding.

    Notes
    -----
    Both run and window tile extents must fit signed 32-bit integers. Invalid
    static controls fail compilation; negative runtime offsets trap before
    writing outputs. Each call prepares its own CUB run table. Use
    :func:`cuda.coop.cutlass.run_length_decode_into` to write a full stream
    while preparing the table once.

    See the :doc:`Run Length Decode visualization
    </python/coop_cutlass/visualizations/run-length-decode>` for windows and
    zero-filled tails.

    Examples
    --------
    Decode a 64-item window starting inside a run, then write the full
    decoded stream at destination offset five. A window extending beyond
    the stream ends with zeros.

    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_qualified_run_length_examples.py
        :language: python
        :start-after: # qualified-run-length-example-begin
        :end-before: # qualified-run-length-example-end
        :dedent: 4
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
    group: BlockGroup,
    run_values: CommonThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    run_lengths: CommonThreadDataLike[_LengthT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    destination: CutlassTensorSample,
    /,
    *,
    decoded_items_per_thread: int,
    destination_offset: IntegralScalar = 0,
    temp_storage: TempStorageLike | None = None,
) -> CompilerIntegerLike:
    """Decode a full stream into a CuTe global-memory tensor.

    The prepared CUB table remains live through all internal decode windows.
    Both run inputs remain unchanged. The destination must not overlap either
    input, and its remaining capacity must hold the full decoded stream.

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
        Each window holds ``block_threads * decoded_items_per_thread`` items;
        the final window writes only the remaining valid items.
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
        outside the decoded interval remain unchanged. The total must fit
        uint32.

    Notes
    -----
    The full stream is decoded; ``destination_offset`` chooses where it is
    written. Offset and capacity validation complete before any output write.
    Negative lengths, misplaced zero padding, and total-size overflow follow
    :func:`cuda.coop.cutlass.run_length_decode`.

    See the :doc:`Run Length Decode visualization
    </python/coop_cutlass/visualizations/run-length-decode>` for bulk output and
    run ordering.

    Examples
    --------
    Write the full decoded stream at destination offset five, alongside
    a register window of the same runs. Every thread receives the total
    decoded length.

    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_qualified_run_length_examples.py
        :language: python
        :start-after: # qualified-run-length-example-begin
        :end-before: # qualified-run-length-example-end
        :dedent: 4
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
