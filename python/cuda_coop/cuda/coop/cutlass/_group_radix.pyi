# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS radix primitives."""

from __future__ import annotations

from typing import Any, overload

import numpy as np
from typing_extensions import TypeVar

from .._typing import IntegerValue, PortableIntegerKey, TraceInteger
from ._temp_storage import TempStorage
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData
from ._thread_group import BlockGroup
from ._typing import CutlassPairValueT

_IntegerKeyT = TypeVar("_IntegerKeyT", bound=PortableIntegerKey)

@overload
def radix_sort_keys(
    group: BlockGroup,
    keys: ThreadData[_IntegerKeyT],
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
) -> ThreadData[_IntegerKeyT]:
    """Return a full-tile radix-sorted CUTLASS block payload.

    ``group`` must be a complete physical block. ``keys`` supplies signed or
    unsigned 32- or 64-bit per-thread keys. ``begin_bit`` and ``end_bit``
    select a half-open interval in CUB's bit-ordered key representation;
    ``end_bit`` defaults to the key width, including when only ``begin_bit`` is
    supplied.
    ``descending`` selects descending order. ``temp_storage`` supplies optional
    caller-owned scratch. The returned ``ThreadData`` preserves the item type
    and count without mutating ``keys``.
    """

@overload
def radix_sort_keys(
    group: BlockGroup,
    keys: _IntegerKeyT,
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
) -> _IntegerKeyT:
    """Return one radix-sorted scalar key per CUTLASS block thread.

    ``group`` must be a complete physical block. ``keys`` supplies one signed
    or unsigned 32- or 64-bit key. ``begin_bit`` selects the least significant
    participating bit; ``end_bit`` is exclusive and defaults to the key width.
    ``descending`` selects descending order. ``temp_storage`` supplies optional
    caller-owned scratch. The scalar result preserves the key type.
    """

@overload
def radix_sort_keys(
    group: BlockGroup,
    keys: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
) -> ThreadData[Any]:
    """Return radix-sorted CUTLASS register-tensor keys as ``ThreadData``.

    ``group`` must be a complete physical block. ``keys`` supplies a qualified
    register-memory Tensor or TensorSSA with a supported integral element type.
    ``begin_bit`` selects the least significant participating bit; ``end_bit``
    is exclusive and defaults to the key width. ``descending`` selects
    descending order. ``temp_storage`` supplies optional caller-owned scratch.
    External compiler dtype tokens necessarily leave the result item type as
    ``Any``.
    """

@overload
def radix_sort_pairs(
    group: BlockGroup,
    keys: ThreadData[_IntegerKeyT],
    values: ThreadData[CutlassPairValueT],
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
) -> tuple[ThreadData[_IntegerKeyT], ThreadData[CutlassPairValueT]]:
    """Return CUTLASS-qualified radix-sorted block key/value payloads.

    ``group`` must be a complete physical block. ``keys`` and ``values`` are
    matching ``ThreadData`` payloads. ``begin_bit`` and ``end_bit`` select a
    half-open interval of the integral key representation. ``descending``
    selects descending order; ``temp_storage`` supplies qualified scratch. The
    result preserves both payload item types and does not mutate either input.
    """

@overload
def radix_sort_pairs(
    group: BlockGroup,
    keys: _IntegerKeyT,
    values: CutlassPairValueT,
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
) -> tuple[_IntegerKeyT, CutlassPairValueT]:
    """Return one radix-sorted CUTLASS key/value pair per block member.

    ``group`` must be a complete physical block. ``keys`` and ``values`` are
    scalar operands. ``begin_bit`` and ``end_bit`` select a half-open interval
    of the integral key representation. ``descending`` selects descending
    order; ``temp_storage`` supplies qualified scratch. The result preserves
    both scalar types.
    """

@overload
def radix_sort_pairs(
    group: BlockGroup,
    keys: CutlassTensorSample | CutlassTensorSSASample,
    values: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
) -> tuple[ThreadData[Any], ThreadData[Any]]:
    """Return radix-sorted CUTLASS register pairs as ``ThreadData`` payloads.

    ``group`` must be a complete physical block. ``keys`` and ``values`` are
    matching register-memory Tensor or TensorSSA payloads. ``begin_bit`` and
    ``end_bit`` select a half-open interval of the integral key representation.
    ``descending`` selects descending order; ``temp_storage`` supplies
    qualified scratch. External compiler dtype tokens necessarily leave both
    result item types as ``Any``.
    """

@overload
def radix_rank(
    group: BlockGroup,
    keys: ThreadData[_IntegerKeyT],
    /,
    *,
    begin_bit: TraceInteger = 0,
    end_bit: TraceInteger | None = None,
    radix_bits: TraceInteger | None = None,
    descending: bool = False,
    exclusive_digit_prefix: ThreadData[int] | ThreadData[np.int32] | None = None,
) -> ThreadData[int]:
    """Return signed 32-bit ranks for a CUTLASS ``ThreadData`` key payload.

    The complete block ranks one trace-static CUB bit-ordered digit without
    mutating ``keys``. ``begin_bit``, ``end_bit``, and ``radix_bits`` are
    trace-time Python or NumPy integers; the selected interval defaults to
    four bits and may contain at most eight bits. ``exclusive_digit_prefix``
    optionally receives signed 32-bit per-digit prefix counters.
    """

@overload
def radix_rank(
    group: BlockGroup,
    keys: _IntegerKeyT,
    /,
    *,
    begin_bit: TraceInteger = 0,
    end_bit: TraceInteger | None = None,
    radix_bits: TraceInteger | None = None,
    descending: bool = False,
    exclusive_digit_prefix: ThreadData[int] | ThreadData[np.int32] | None = None,
) -> int:
    """Return one signed 32-bit rank for a CUTLASS scalar integer key.

    The specialization controls and optional prefix output have the same
    meaning as for the ``ThreadData`` overload.
    """

@overload
def radix_rank(
    group: BlockGroup,
    keys: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    begin_bit: TraceInteger = 0,
    end_bit: TraceInteger | None = None,
    radix_bits: TraceInteger | None = None,
    descending: bool = False,
    exclusive_digit_prefix: ThreadData[int] | ThreadData[np.int32] | None = None,
) -> ThreadData[int]:
    """Return signed 32-bit ranks for a CUTLASS register tensor.

    The compiler validates the opaque tensor element type against the i32,
    u32, i64, and u64 provider set. The result is flattened ``ThreadData``
    with the same per-thread item count. The specialization controls and
    optional prefix output follow the other qualified overloads.
    """

__all__ = [
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_pairs",
]
