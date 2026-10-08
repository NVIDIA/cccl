# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose block differences and run-boundary flags for compiled kernels.

Adjacent Difference subtracts neighboring values. Discontinuity identifies
unequal neighbors as run heads or tails. Both preserve the source payload and
return separate blocked results. Decorators register each function so a
supported compiler can recognize calls to it. Ordinary Python calls raise
a context error. Custom binary operations use the qualified API.
"""

from __future__ import annotations

from ..thread_group import CoopCompilerContextRequiredError
from ._dispatch import (
    _common_group_operation,
)
from ._payload import (
    TempStorageLike,
    ThreadDataLike,
)

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
from typing import Literal, TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    IntegerValue,
)

from .thread_group import BlockGroup

_T = TypeVar("_T", bound=CommonNumericScalar)


@_common_group_operation("adjacent_difference", group_kinds=("block",))
def adjacent_difference(
    group: BlockGroup,
    values: CommonThreadDataLike[_T],
    /,
    *,
    direction: Literal["left", "right"] = "left",
    valid_items: IntegerValue | None = None,
    tile_predecessor_item: CommonNumericScalar | None = None,
    tile_successor_item: CommonNumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_T]:
    """Return blocked neighbor differences without modifying the input.

    Parameters
    ----------
    group : ThreadGroup
        A complete ``this_block()`` group. Every member participates in the
        same call; multidimensional blocks use linear thread-rank order.
    values : ThreadDataLike
        Readable fixed-size payload in blocked order. Signed and unsigned
        8-, 16-, 32-, and 64-bit integers, float32, and float64 are supported.
        Initialize every input slot, including any invalid suffix.
    direction : {"left", "right"}, optional
        Compile-time choice, default ``"left"``. Subtract the previous item
        from the current item for left differences, or the next item from
        the current item for right differences.
    valid_items : integer, optional
        Block-uniform count in ``[0, block_size * items_per_thread]``.
        Omit it to process the full tile. The suffix beyond this count is
        copied unchanged. For right differences, this argument cannot be
        combined with ``tile_successor_item``.
    tile_predecessor_item, tile_successor_item : scalar, optional
        Block-uniform neighbor outside the tile. A runtime or NumPy scalar
        must match the input dtype. Finite Python int or float literals take
        the input dtype when in range; conversion to float may round. A float
        literal cannot become an integer.
        Left differences accept only a predecessor; right differences accept
        only a successor. Without that neighbor, the boundary input is
        copied unchanged.
    temp_storage : TempStorageLike, optional
        Explicit block scratch descriptor. Omit it for automatic storage.
        With ``auto_sync=False``, synchronize the block before reusing the
        descriptor in another collective.

    Returns
    -------
    ThreadDataLike
        A fresh blocked payload with the input dtype and extent. The input
        and the returned invalid suffix retain their original values.

    Notes
    -----
    Subtraction uses the input dtype. Use
    :func:`cuda.coop.numba_mlir.adjacent_difference` for a custom binary
    operator. The CUB counterpart is ``cub::BlockAdjacentDifference``.

    Examples
    --------
    Delta-encode an array across three blocks with Numba-CUDA-MLIR. Each
    block supplies the preceding tile's last input value, so differences
    remain continuous across tile boundaries.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_neighbors.py
        :language: python
        :start-after: # adjacent-difference-example-begin
        :end-before: # adjacent-difference-example-end
        :dedent: 4
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.adjacent_difference must be called from a supported "
        "GPU kernel."
    )


@_common_group_operation("discontinuity", group_kinds=("block",))
def discontinuity(
    group: BlockGroup,
    values: CommonThreadDataLike[_T],
    /,
    *,
    mode: Literal["heads", "tails", "heads_and_tails"] = "heads",
    tile_predecessor_item: CommonNumericScalar | None = None,
    tile_successor_item: CommonNumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> (
    ThreadDataLike[np.int32]
    | tuple[ThreadDataLike[np.int32], ThreadDataLike[np.int32]]
):
    """Flag unequal adjacent items in a full, blocked block tile.

    Parameters
    ----------
    group : ThreadGroup
        A complete ``this_block()`` group. Every member participates in the
        same call; multidimensional blocks use linear thread-rank order.
    values : ThreadDataLike
        Readable fixed-size payload in blocked order. Signed and unsigned
        8-, 16-, 32-, and 64-bit integers, float32, and float64 are supported.
    mode : {"heads", "tails", "heads_and_tails"}, optional
        Compile-time result selection, default ``"heads"``. Heads compare
        the previous item with the current item; tails compare the current
        item with the next item. Unequal neighbors produce one, otherwise
        zero.
    tile_predecessor_item, tile_successor_item : scalar, optional
        Block-uniform neighbor outside the tile. A runtime or NumPy scalar
        must match the input dtype. Finite Python int or float literals take
        the input dtype when in range; conversion to float may round. A float
        literal cannot become an integer.
        Heads accept a predecessor, tails accept a successor, and
        ``"heads_and_tails"`` accepts both. Without a predecessor the first
        head is one; without a successor the last tail is one.
    temp_storage : TempStorageLike, optional
        Explicit block scratch descriptor. Omit it for automatic storage.
        With ``auto_sync=False``, synchronize the block before reusing the
        descriptor in another collective.

    Returns
    -------
    ThreadDataLike or tuple of ThreadDataLike
        A fresh int32 payload for heads or tails, or ``(heads, tails)`` for
        ``"heads_and_tails"``. Each payload has the input extent and blocked
        layout. The input is preserved.

    Notes
    -----
    Partial tiles are unsupported. Padding participates in comparisons and
    can change the last valid tail. Use
    :func:`cuda.coop.numba_mlir.discontinuity` for a custom binary predicate.
    The CUB counterpart is ``cub::BlockDiscontinuity``.

    Examples
    --------
    Assign a label to each run of equal keys with Numba-CUDA-MLIR. Scanning
    the head flags produces labels that restart at zero in each block tile.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_neighbors.py
        :language: python
        :start-after: # discontinuity-example-begin
        :end-before: # discontinuity-example-end
        :dedent: 4
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.discontinuity must be called from a supported GPU kernel."
    )


__all__ = ["adjacent_difference", "discontinuity"]
