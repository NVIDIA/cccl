# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable block radix ordering operations."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from ..block.radix import make_radix_bit_range
from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup
from ._dispatch import (
    _portable_group_operation,
)
from ._payload import (
    TempStorageLike,
)


def _radix_bounds(operation, key_width, begin_bit, end_bit, radix_bits=None):
    for name, value in (
        ("begin_bit", begin_bit),
        ("end_bit", end_bit),
        ("radix_bits", radix_bits),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral)
        ):
            raise TypeError(
                f"cuda.coop.{operation} {name} must be a compile-time integer"
            )
    if radix_bits is not None and radix_bits <= 0:
        raise ValueError("radix_bits must be positive")
    if end_bit is None:
        end_bit = (
            begin_bit + (4 if radix_bits is None else radix_bits)
            if operation == "radix_rank"
            else key_width
        )
    if radix_bits is not None and end_bit - begin_bit != radix_bits:
        raise ValueError("radix_bits must match end_bit - begin_bit")
    make_radix_bit_range(begin_bit=begin_bit, end_bit=end_bit, bit_width=key_width)
    if operation == "radix_rank" and end_bit - begin_bit > 8:
        raise ValueError("radix_rank bit width must be <= 8")
    return int(begin_bit), int(end_bit)


@_portable_group_operation("radix_sort_keys", group_kinds=("block",))
def radix_sort_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Return stable, blocked radix-sorted integral keys without mutation.

    The complete block participates. Keys are fixed-size ThreadData with
    32- or 64-bit signed or unsigned integer elements. The half-open bit
    interval selects CUB's ordered representation, including the inverted
    sign bit for signed integers; omitted end_bit selects the key width.
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.radix_sort_keys must be called from a supported GPU kernel."
    )


@_portable_group_operation("radix_sort_pairs", group_kinds=("block",))
def radix_sort_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
) -> tuple[Any, Any]:
    """Return stable sorted keys and associated numeric values without mutation.

    Both ThreadData payloads must have the same extent. Equal selected key
    digits retain flattened blocked input order, including descending sorts.
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.radix_sort_pairs must be called from a supported GPU kernel."
    )


@_portable_group_operation("radix_rank", group_kinds=("block",))
def radix_rank(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
) -> Any:
    """Return shape-preserving int32 ranks without mutating integral keys.

    Rank the selected digit stably across a complete block. The interval
    defaults to four bits starting at begin_bit and may contain at most eight
    bits. Signed keys use the same sign-bit transformation as radix sort.
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.radix_rank must be called from a supported GPU kernel."
    )


__all__ = ["radix_rank", "radix_sort_keys", "radix_sort_pairs"]
