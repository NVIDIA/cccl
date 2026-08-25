# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable TopK family."""

from typing_extensions import TypeVar

from cuda.coop._typing import (
    IntegerValue,
    PortableIntegerKey,
    PortableNumericScalar,
    TempStorageLike,
    ThreadDataLike,
    ValidItems,
)

from .thread_group import BlockGroup

_PortableNumericT = TypeVar("_PortableNumericT", bound=PortableNumericScalar)
_IntegerKeyT = TypeVar("_IntegerKeyT", bound=PortableIntegerKey)

def topk_max_keys(
    group: BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    k: IntegerValue,
    /,
    *,
    valid_items: ValidItems | None = None,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_IntegerKeyT]:
    """Return the largest keys in an unordered, shape-preserving prefix.

    ``group`` must be a complete one-dimensional physical block. ``keys`` is
    a fixed-size payload of Python integers or signed or unsigned 32- or
    64-bit NumPy or compiler integers. ``k`` and ``valid_items`` are uniform
    integer values satisfying ``1 <= k <= valid_items``; omitting
    ``valid_items`` selects the full block tile. ``begin_bit`` and ``end_bit``
    select a nonempty half-open interval in CUB's bit-ordered key
    representation, with ``end_bit=None`` selecting the key width. Only the
    first ``k`` flattened blocked positions are defined. That prefix is not
    sorted, ties do not expand it, and the remaining positions are undefined.
    The result preserves the input item type and count without mutating
    ``keys``. ``temp_storage`` supplies optional reusable scratch.
    """

def topk_min_keys(
    group: BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    k: IntegerValue,
    /,
    *,
    valid_items: ValidItems | None = None,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_IntegerKeyT]:
    """Return the smallest keys in an unordered, shape-preserving prefix.

    ``group`` must be a complete one-dimensional physical block. ``keys`` is
    a fixed-size payload of Python integers or signed or unsigned 32- or
    64-bit NumPy or compiler integers. ``k`` and ``valid_items`` are uniform
    integer values satisfying ``1 <= k <= valid_items``; omitting
    ``valid_items`` selects the full block tile. ``begin_bit`` and ``end_bit``
    select a nonempty half-open interval in CUB's bit-ordered key
    representation, with ``end_bit=None`` selecting the key width. Only the
    first ``k`` flattened blocked positions are defined. That prefix is not
    sorted, ties do not expand it, and the remaining positions are undefined.
    The result preserves the input item type and count without mutating
    ``keys``. ``temp_storage`` supplies optional reusable scratch.
    """

def topk_max_pairs(
    group: BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    values: ThreadDataLike[_PortableNumericT],
    k: IntegerValue,
    /,
    *,
    valid_items: ValidItems | None = None,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[
    ThreadDataLike[_IntegerKeyT],
    ThreadDataLike[_PortableNumericT],
]:
    """Return the largest-key pairs in an unordered prefix.

    Exactly the first ``k`` flattened blocked pairs are defined. Values remain
    attached to their keys and the remaining positions are undefined.
    ``temp_storage`` supplies optional reusable scratch.
    """

def topk_min_pairs(
    group: BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    values: ThreadDataLike[_PortableNumericT],
    k: IntegerValue,
    /,
    *,
    valid_items: ValidItems | None = None,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[
    ThreadDataLike[_IntegerKeyT],
    ThreadDataLike[_PortableNumericT],
]:
    """Return the smallest-key pairs in an unordered prefix.

    Exactly the first ``k`` flattened blocked pairs are defined. Values remain
    attached to their keys and the remaining positions are undefined.
    ``temp_storage`` supplies optional reusable scratch.
    """
