# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable radix sort and rank."""

from typing_extensions import TypeVar

from cuda.coop._typing import (
    IntegerValue,
    PortableIntegerKey,
    PortableNumericScalar,
    TempStorageLike,
    ThreadDataLike,
    TraceInteger,
)

from .thread_group import BlockGroup

_PortableNumericT = TypeVar("_PortableNumericT", bound=PortableNumericScalar)
_IntegerKeyT = TypeVar("_IntegerKeyT", bound=PortableIntegerKey)

def radix_sort_keys(
    group: BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_IntegerKeyT]:
    """Return a full-tile radix-sorted block payload without mutating ``keys``.

    ``group`` must be a complete physical block. ``keys`` must be a fixed-size
    ``ThreadDataLike`` payload of signed or unsigned 32- or 64-bit Python,
    NumPy, or compiler integer values. ``begin_bit`` and ``end_bit`` select a
    half-open interval in CUB's bit-ordered key representation. ``end_bit``
    defaults to the key width, including when only ``begin_bit`` is supplied.
    ``descending`` selects descending order. ``temp_storage`` supplies optional
    caller-owned scratch. The returned payload preserves the input item type
    and item count.
    """

def radix_sort_pairs(
    group: BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    values: ThreadDataLike[_PortableNumericT],
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
) -> tuple[
    ThreadDataLike[_IntegerKeyT],
    ThreadDataLike[_PortableNumericT],
]:
    """Return radix-sorted block key/value payloads without mutation.

    Keys and values have matching item counts and retain their independent
    types. Values move with keys across the selected bit-ordered range.
    """

def radix_rank(
    group: BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    /,
    *,
    begin_bit: TraceInteger = 0,
    end_bit: TraceInteger | None = None,
    radix_bits: TraceInteger | None = None,
    descending: bool = False,
) -> ThreadDataLike[int]:
    """Return shape-preserving signed 32-bit ranks for one radix digit.

    ``group`` must be a complete physical block. ``keys`` must be a fixed-size
    payload of signed or unsigned 32- or 64-bit Python, NumPy, or compiler
    integer values. ``begin_bit``, ``end_bit``, and ``radix_bits`` are
    trace-time Python or NumPy integers. The selected half-open CUB bit-ordered
    interval defaults to four bits and may contain at most eight bits. Equal
    digits retain flattened blocked input order. The operation leaves ``keys``
    unchanged. Use a qualified import for scalar/register inputs or an
    exclusive digit-prefix side output.
    """
