# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable discontinuity family."""

from typing import Literal, overload

from typing_extensions import TypeVar

from cuda.coop._typing import PortableNumericScalar, TempStorageLike, ThreadDataLike

from .thread_group import BlockGroup

_PortableNumericT = TypeVar("_PortableNumericT", bound=PortableNumericScalar)

@overload
def discontinuity(
    group: BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: Literal["heads"] = "heads",
    tile_predecessor_item: _PortableNumericT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[int]:
    """Return block-wide signed 32-bit head flags.

    ``group`` must be a complete physical block and ``value`` must be a
    fixed-size per-thread payload. ``mode`` is ``"heads"`` and compares each
    item with its predecessor. The returned flag payload has the same shape as
    ``value``, and ``value`` is not mutated.

    ``tile_predecessor_item`` supplies the head boundary and must match the
    payload item type; without it, the first head is set.
    ``tile_successor_item`` stays ``None``. ``temp_storage`` supplies optional
    caller-owned scratch. The portable root uses built-in inequality.
    """

@overload
def discontinuity(
    group: BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: Literal["tails"],
    tile_predecessor_item: None = None,
    tile_successor_item: _PortableNumericT | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[int]:
    """Return block-wide signed 32-bit tail flags.

    ``group`` must be a complete physical block and ``value`` must be a
    fixed-size per-thread payload. ``mode`` is ``"tails"`` and compares each
    item with its successor. The returned flag payload has the same shape as
    ``value``, and ``value`` is not mutated.

    ``tile_predecessor_item`` stays ``None``. ``tile_successor_item`` supplies
    the tail boundary and must match the payload item type; without it, the
    final tail is set. ``temp_storage`` supplies optional caller-owned scratch.
    The portable root uses built-in inequality and intentionally excludes the
    qualified ``"heads_and_tails"`` pair-returning mode.
    """
