# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common block Adjacent Difference and Discontinuity APIs."""

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup
from ._dispatch import (
    _portable_group_operation,
)
from ._payload import (
    TempStorageLike,
    ThreadDataLike,
    _ReadableThreadDataLike,
)


@_portable_group_operation("adjacent_difference", group_kinds=("block",))
def adjacent_difference(
    group: ThreadGroup,
    values: _ReadableThreadDataLike[Any],
    /,
    *,
    direction: str = "left",
    valid_items: object = None,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[Any]:
    """Return blocked neighbor differences without modifying the input.

    A complete block processes fixed-size numeric ThreadData. Left subtracts
    the previous element from the current element; right subtracts the next
    element. A missing neighbor leaves the boundary input unchanged unless
    the matching tile boundary scalar is supplied. Boundary scalars must be
    uniform across the block and match the input dtype.

    valid_items is a uniform count in [0, block size * items per thread].
    The invalid suffix is copied unchanged. Right partial tiles cannot use
    tile_successor_item. All input slots must be initialized, including the
    suffix. Temporary storage is automatic unless temp_storage is supplied.
    Use the qualified API for a custom binary difference operator.
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.adjacent_difference must be called from a supported GPU kernel."
    )


@_portable_group_operation("discontinuity", group_kinds=("block",))
def discontinuity(
    group: ThreadGroup,
    values: _ReadableThreadDataLike[Any],
    /,
    *,
    mode: str = "heads",
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[Any] | tuple[ThreadDataLike[Any], ThreadDataLike[Any]]:
    """Flag unequal adjacent items in a full, blocked block tile.

    Return int32 ThreadData for heads or tails, or (heads, tails) for
    heads_and_tails. Results have the input extent; the input is preserved.
    An absent tile predecessor forces the first head to one; an absent
    successor forces the last tail to one. Supplied boundary scalars must
    match the input dtype and be uniform across the block. Only the boundary
    arguments relevant to the selected mode are accepted.

    Partial tiles are unsupported: padding becomes part of the tile and can
    change its flags, including the last valid tail. Storage is automatic
    unless temp_storage is supplied. Use the qualified API for a custom
    binary predicate.
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.discontinuity must be called from a supported GPU kernel."
    )


__all__ = ["adjacent_difference", "discontinuity"]
