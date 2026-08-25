# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable merge-sort family."""

from typing import overload

from typing_extensions import TypeVar

from cuda.coop._typing import (
    PortableIntegerKey,
    PortableNumericScalar,
    TempStorageLike,
    ThreadDataLike,
    ValidItems,
)

from .thread_group import MemoryGroup

_PortableNumericT = TypeVar("_PortableNumericT", bound=PortableNumericScalar)
_IntegerKeyT = TypeVar("_IntegerKeyT", bound=PortableIntegerKey)

@overload
def merge_sort_keys(
    group: MemoryGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_IntegerKeyT]:
    """Return a fully merge-sorted integral payload without mutating ``keys``.

    ``group`` must be a complete physical block, physical warp, or logical warp;
    a block must contain a power-of-two number of threads. ``keys`` must be a fixed-size
    ``ThreadDataLike`` payload of Python, NumPy, or compiler integer values. The
    returned payload preserves the input item type and item count.
    """

@overload
def merge_sort_keys(
    group: MemoryGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: ValidItems,
    oob_default: _IntegerKeyT | int,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_IntegerKeyT]:
    """Return a partial-tile merge-sorted integral payload.

    ``valid_items`` and ``oob_default`` must be supplied together. The sentinel
    must have the matching key dtype or be a plain Python integer representable
    in that dtype. A block must contain a power-of-two number of threads. The
    sentinel must sort after every valid key: greater for ascending order and
    less for descending order. Only the valid sorted prefix is defined; the
    returned payload still preserves the input item type and item count. Use a
    qualified import for custom comparators or backend-specific group and
    payload forms.
    """

@overload
def merge_sort_pairs(
    group: MemoryGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    values: ThreadDataLike[_PortableNumericT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[
    ThreadDataLike[_IntegerKeyT],
    ThreadDataLike[_PortableNumericT],
]:
    """Return fully merge-sorted key/value payloads without mutation.

    Keys determine ordering and each numeric value remains attached to its key.
    Both result payloads preserve their independent item types and common item
    count. Equal-key ordering is unspecified.
    """

@overload
def merge_sort_pairs(
    group: MemoryGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    values: ThreadDataLike[_PortableNumericT],
    /,
    *,
    descending: bool = False,
    valid_items: ValidItems,
    oob_default: _IntegerKeyT | int,
    temp_storage: TempStorageLike | None = None,
) -> tuple[
    ThreadDataLike[_IntegerKeyT],
    ThreadDataLike[_PortableNumericT],
]:
    """Return a partial-tile merge-sorted key/value prefix.

    ``valid_items`` and ``oob_default`` are supplied together. The sentinel
    must have the matching key dtype or be a plain Python integer representable
    in that dtype. Only the valid sorted prefix is defined; key/value
    association is preserved.
    """
