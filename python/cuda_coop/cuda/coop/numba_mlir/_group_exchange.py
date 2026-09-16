# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Exchange marker for Numba-CUDA-MLIR."""

from __future__ import annotations

from typing import Any

from .._core.api._payload import ThreadDataLike, _ReadableThreadDataLike
from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation(
    "exchange",
    family_module="cuda.coop.numba_mlir._compiler._group_exchange",
)
def exchange(
    group: ThreadGroup,
    value: _ReadableThreadDataLike[Any],
    /,
    *,
    mode: Any = "striped_to_blocked",
    ranks: _ReadableThreadDataLike[Any] | None = None,
    valid_flags: _ReadableThreadDataLike[Any] | None = None,
    warp_time_slicing: bool = False,
) -> ThreadDataLike[Any]:
    """Exchange per-thread values, including ranked block scatters.

    See :func:`cuda.coop.exchange` for shared group requirements, layout
    definitions, dtypes, and examples. The qualified API adds local arrays,
    block scatter modes, and the options below. Every group member must call
    the collective, including members whose scatter items are all invalid.

    Parameters
    ----------
    value : cuda.coop.ThreadDataLike or local array
        Also accepts a one-dimensional ``cuda.local.array`` with a fixed
        extent. All members must use the same dtype and extent.
    mode : str, optional
        Compile-time conversion, default ``"striped_to_blocked"``. Blocks
        also support ``"blocked_to_warp_striped"`` and
        ``"warp_striped_to_blocked"``, with striping within each physical
        warp; these require a block size divisible by 32. Block scatter modes
        are ``"scatter_to_blocked"``, ``"scatter_to_striped"``,
        ``"scatter_to_striped_guarded"``, and ``"scatter_to_striped_flagged"``.
        Warp groups support only ``"striped_to_blocked"`` and
        ``"blocked_to_striped"``.
    ranks : cuda.coop.ThreadDataLike or local array, optional
        Destination ranks for scatter modes, required for every scatter mode
        and rejected for layout conversions. Must have a signed integer
        dtype and the same per-thread extent as ``value``. Valid ranks must
        be distinct across the block and lie in ``[0, block_tile_size)``.
        The guarded mode skips negative ranks. Other scatter modes require
        every participating item's rank to be in range. ``None`` is the
        default.
    valid_flags : cuda.coop.ThreadDataLike or local array, optional
        Per-item nonzero flags for ``"scatter_to_striped_flagged"``; required
        for that mode and rejected for all others. Must have an integer,
        non-boolean dtype and the same extent as ``value``. Only flagged items
        scatter; their ranks must be distinct and in range. ``None`` is the
        default.
    warp_time_slicing : bool, optional
        Compile-time option, default ``False``. Reuse block exchange scratch
        across warps to reduce shared memory usage at the cost of additional
        synchronization. Requires a block; guarded and flagged scatter modes
        do not support it.

    Returns
    -------
    cuda.coop.ThreadDataLike or local array
        New writable payload with the input dtype and extent in the requested
        output layout. Inputs and ranks are preserved. A scatter destination
        with no valid input is undefined and must not be read before it is
        initialized.

    Notes
    -----
    The implementation manages :ref:`temporary storage <coop-temp-storage>`
    automatically. See :ref:`coop-thread-data` for payloads and
    :ref:`coop-thread-groups` for participation requirements.

    See Also
    --------
    cuda.coop.exchange
        Shared blocked and striped layout conversions.
    :cpp:struct:`cub::BlockExchange`, :cpp:struct:`cub::WarpExchange`
        C++ exchange primitives.

    Examples
    --------
    Reverse a block tile by assigning each input element its destination rank.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_movement_examples.py
        :language: python
        :start-after: # scatter-example-begin
        :end-before: # scatter-example-end
        :dedent: 4
    """

    return group_primitive_marker(
        "exchange",
        group,
        value,
        mode=mode,
        ranks=ranks,
        valid_flags=valid_flags,
        warp_time_slicing=warp_time_slicing,
    )


__all__ = ["exchange"]
