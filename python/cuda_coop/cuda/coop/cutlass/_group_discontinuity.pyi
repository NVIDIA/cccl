# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS discontinuity."""

from __future__ import annotations

from typing import Literal, TypeAlias, overload

from .._typing import PortableNumericScalar
from ._temp_storage import TempStorage
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData
from ._thread_group import BlockGroup
from ._typing import CutlassNumericT, ScalarValueT

_FlagOperator: TypeAlias = Literal["!=", "ne", "not_equal"]

@overload
def discontinuity(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    mode: Literal["heads"] = "heads",
    tile_predecessor_item: CutlassNumericT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    flag_op: _FlagOperator | None = None,
) -> ThreadData[int]:
    """Return one CUTLASS signed 32-bit head-flag payload.

    ``group`` must be a complete physical block and ``value`` must be a
    fixed-size per-thread payload. ``mode`` is ``"heads"`` and the result
    preserves its shape without mutating ``value``. ``tile_predecessor_item``
    supplies a same-typed head boundary; ``tile_successor_item`` stays ``None``.
    ``temp_storage`` supplies scratch, and ``flag_op`` selects built-in
    inequality.
    """

@overload
def discontinuity(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    mode: Literal["tails"],
    tile_predecessor_item: None = None,
    tile_successor_item: CutlassNumericT | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: _FlagOperator | None = None,
) -> ThreadData[int]:
    """Return one CUTLASS signed 32-bit tail-flag payload.

    ``group`` must be a complete physical block and ``value`` must be a
    fixed-size per-thread payload. ``mode`` is ``"tails"`` and the result
    preserves its shape without mutating ``value``. ``tile_predecessor_item``
    stays ``None``; ``tile_successor_item`` supplies a same-typed tail boundary.
    ``temp_storage`` supplies scratch, and ``flag_op`` selects built-in
    inequality.
    """

@overload
def discontinuity(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    mode: Literal["heads_and_tails"],
    tile_predecessor_item: CutlassNumericT | None = None,
    tile_successor_item: CutlassNumericT | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: _FlagOperator | None = None,
) -> tuple[ThreadData[int], ThreadData[int]]:
    """Return CUTLASS signed 32-bit head and tail flag payloads.

    ``group`` must be a complete physical block, ``value`` must be a fixed-size
    per-thread payload, and ``mode`` must be ``"heads_and_tails"``. The two
    results preserve the payload shape without mutating ``value``.
    ``tile_predecessor_item`` and ``tile_successor_item`` supply same-typed head
    and tail boundaries. ``temp_storage`` supplies optional caller-owned
    scratch, ``flag_op`` selects built-in inequality.
    """

@overload
def discontinuity(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    mode: Literal["heads"] = "heads",
    tile_predecessor_item: PortableNumericScalar | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    flag_op: _FlagOperator | None = None,
) -> ThreadData[int]:
    """Return head flags for a CUTLASS register tensor.

    ``group`` is a complete block, ``value`` supplies a static rmem or SSA
    payload, and ``mode`` is ``"heads"``. ``tile_predecessor_item`` provides
    the head edge value and ``tile_successor_item`` stays ``None``.
    ``temp_storage`` supplies scratch, and ``flag_op`` selects built-in
    inequality. The result preserves the flattened item count.
    """

@overload
def discontinuity(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    mode: Literal["tails"],
    tile_predecessor_item: None = None,
    tile_successor_item: PortableNumericScalar | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: _FlagOperator | None = None,
) -> ThreadData[int]:
    """Return tail flags for a CUTLASS register tensor.

    ``group`` is a complete block, ``value`` supplies a static rmem or SSA
    payload, and ``mode`` is ``"tails"``. ``tile_predecessor_item`` stays
    ``None`` and ``tile_successor_item`` provides the tail edge value.
    ``temp_storage`` supplies scratch, and ``flag_op`` selects built-in
    inequality. The result preserves the flattened item count.
    """

@overload
def discontinuity(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    mode: Literal["heads_and_tails"],
    tile_predecessor_item: PortableNumericScalar | None = None,
    tile_successor_item: PortableNumericScalar | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: _FlagOperator | None = None,
) -> tuple[ThreadData[int], ThreadData[int]]:
    """Return head and tail flags for a CUTLASS register tensor.

    ``group`` is a complete block, ``value`` supplies a static rmem or SSA
    payload, and ``mode`` requests both signed 32-bit flag projections.
    ``tile_predecessor_item`` and ``tile_successor_item`` provide edge values.
    ``temp_storage`` supplies scratch, and ``flag_op`` selects built-in
    inequality. Both results preserve the flattened item count.
    """

@overload
def discontinuity(
    group: BlockGroup,
    value: ScalarValueT,
    /,
    *,
    mode: Literal["heads"] = "heads",
    tile_predecessor_item: ScalarValueT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    flag_op: _FlagOperator | None = None,
) -> int:
    """Return one qualified scalar head flag.

    ``group`` must be a complete block and ``value`` is the calling thread's
    scalar item. ``mode`` is ``"heads"``. ``tile_predecessor_item`` supplies a
    same-typed head boundary and ``tile_successor_item`` stays ``None``.
    ``temp_storage`` supplies scratch, and ``flag_op`` selects built-in
    inequality.
    """

@overload
def discontinuity(
    group: BlockGroup,
    value: ScalarValueT,
    /,
    *,
    mode: Literal["tails"],
    tile_predecessor_item: None = None,
    tile_successor_item: ScalarValueT | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: _FlagOperator | None = None,
) -> int:
    """Return one qualified scalar tail flag.

    ``group`` must be a complete block and ``value`` is the calling thread's
    scalar item. ``mode`` is ``"tails"``. ``tile_predecessor_item`` stays
    ``None`` and ``tile_successor_item`` supplies a same-typed tail boundary.
    ``temp_storage`` supplies scratch, and ``flag_op`` selects built-in
    inequality.
    """

@overload
def discontinuity(
    group: BlockGroup,
    value: ScalarValueT,
    /,
    *,
    mode: Literal["heads_and_tails"],
    tile_predecessor_item: ScalarValueT | None = None,
    tile_successor_item: ScalarValueT | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: _FlagOperator | None = None,
) -> tuple[int, int]:
    """Return qualified scalar head and tail discontinuity flags.

    ``group`` must be a complete block, ``value`` is the calling thread's
    scalar item, and ``mode`` selects both flags. ``tile_predecessor_item`` and
    ``tile_successor_item`` supply same-typed boundaries. ``temp_storage``
    supplies scratch, ``flag_op`` selects built-in inequality.
    """

__all__ = [
    "discontinuity",
]
