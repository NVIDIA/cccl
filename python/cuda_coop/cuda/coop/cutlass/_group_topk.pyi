# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS TopK primitives."""

from __future__ import annotations

from typing import Any, overload

from .._typing import _IntegerValue as _IntegerValue
from .._typing import _ValidItems as _ValidItems
from ._types import TempStorage, ThreadData
from ._types import _BlockGroup as _BlockGroup
from ._types import _CutlassTensorSample as _CutlassTensorSample
from ._types import _CutlassTensorSSASample as _CutlassTensorSSASample
from ._types import _CutlassTopKKeyT as _CutlassTopKKeyT
from ._types import _CutlassTopKValueT as _CutlassTopKValueT

@overload
def topk_max_keys(
    group: _BlockGroup,
    keys: ThreadData[_CutlassTopKKeyT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadData[_CutlassTopKKeyT]:
    """Select the largest keys from a CUTLASS ``ThreadData`` block tile.

    ``group`` is a complete one-dimensional physical block. ``keys`` may use
    any CUTLASS TopK provider dtype. Uniform ``k`` and ``valid_items`` satisfy
    ``1 <= k <= valid_items``; omitting ``valid_items`` selects the full tile.
    Uniform ``begin_bit`` and ``end_bit`` select a nonempty half-open interval,
    with ``end_bit=None`` selecting the key width. Only the first ``k``
    flattened blocked positions are defined; they are unordered, ties do not
    expand the prefix, and the tail is undefined. The new payload preserves the
    item type and count without mutating ``keys``. ``temp_storage`` optionally
    reuses caller-owned block scratch.
    """

@overload
def topk_max_keys(
    group: _BlockGroup,
    keys: _CutlassTopKKeyT,
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> _CutlassTopKKeyT:
    """Select the largest key from one scalar per CUTLASS block member.

    ``group``, ``k``, ``valid_items``, ``begin_bit``, and ``end_bit`` follow the
    qualified payload contract. Only members in the unordered first-``k`` block
    prefix receive defined results. The scalar result preserves the type of
    ``keys``. ``temp_storage`` optionally reuses caller-owned block scratch.
    """

@overload
def topk_max_keys(
    group: _BlockGroup,
    keys: _CutlassTensorSample | _CutlassTensorSSASample,
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadData[Any]:
    """Select the largest keys from a CUTLASS register tensor.

    ``keys`` supplies the register payload. ``group``, ``k``, ``valid_items``,
    ``begin_bit``, and ``end_bit`` follow the qualified payload contract. The
    register payload is adapted to ``ThreadData`` without mutation. External
    compiler element types necessarily leave the result item type as ``Any``.
    ``temp_storage`` optionally reuses caller-owned block scratch.
    """

@overload
def topk_min_keys(
    group: _BlockGroup,
    keys: ThreadData[_CutlassTopKKeyT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadData[_CutlassTopKKeyT]:
    """Select the smallest keys from a CUTLASS ``ThreadData`` block tile.

    ``group`` is a complete one-dimensional physical block. ``keys`` may use
    any CUTLASS TopK provider dtype. Uniform ``k`` and ``valid_items`` satisfy
    ``1 <= k <= valid_items``; omitting ``valid_items`` selects the full tile.
    Uniform ``begin_bit`` and ``end_bit`` select a nonempty half-open interval,
    with ``end_bit=None`` selecting the key width. Only the first ``k``
    flattened blocked positions are defined; they are unordered, ties do not
    expand the prefix, and the tail is undefined. The new payload preserves the
    item type and count without mutating ``keys``. ``temp_storage`` optionally
    reuses caller-owned block scratch.
    """

@overload
def topk_min_keys(
    group: _BlockGroup,
    keys: _CutlassTopKKeyT,
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> _CutlassTopKKeyT:
    """Select the smallest key from one scalar per CUTLASS block member.

    ``group``, ``k``, ``valid_items``, ``begin_bit``, and ``end_bit`` follow the
    qualified payload contract. Only members in the unordered first-``k`` block
    prefix receive defined results. The scalar result preserves the type of
    ``keys``. ``temp_storage`` optionally reuses caller-owned block scratch.
    """

@overload
def topk_min_keys(
    group: _BlockGroup,
    keys: _CutlassTensorSample | _CutlassTensorSSASample,
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadData[Any]:
    """Select the smallest keys from a CUTLASS register tensor.

    ``keys`` supplies the register payload. ``group``, ``k``, ``valid_items``,
    ``begin_bit``, and ``end_bit`` follow the qualified payload contract. The
    register payload is adapted to ``ThreadData`` without mutation. External
    compiler element types necessarily leave the result item type as ``Any``.
    ``temp_storage`` optionally reuses caller-owned block scratch.
    """

@overload
def topk_max_pairs(
    group: _BlockGroup,
    keys: ThreadData[_CutlassTopKKeyT],
    values: ThreadData[_CutlassTopKValueT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[ThreadData[_CutlassTopKKeyT], ThreadData[_CutlassTopKValueT]]:
    """Select largest-key CUTLASS ``ThreadData`` pairs without mutation.

    ``group`` is a complete one-dimensional physical block. ``keys`` and
    ``values`` have matching positive item counts and supported provider
    dtypes. Uniform ``k``, ``valid_items``, ``begin_bit``, and ``end_bit``
    follow the qualified keys-only contract. Only the first ``k`` unordered
    pairs are defined, and each value remains attached to its key. Both result
    item types are preserved. ``temp_storage`` optionally reuses caller-owned
    block scratch.
    """

@overload
def topk_max_pairs(
    group: _BlockGroup,
    keys: _CutlassTopKKeyT,
    values: _CutlassTopKValueT,
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[_CutlassTopKKeyT, _CutlassTopKValueT]:
    """Select largest-key pairs from one scalar pair per block member.

    ``keys`` and ``values`` supply one scalar pair. ``group``, ``k``,
    ``valid_items``, ``begin_bit``, and ``end_bit`` follow the qualified
    keys-only contract. Only the unordered first-``k`` block positions are
    defined. Both scalar result types are preserved and values remain attached
    to their keys. ``temp_storage`` optionally reuses caller-owned block scratch.
    """

@overload
def topk_max_pairs(
    group: _BlockGroup,
    keys: _CutlassTensorSample | _CutlassTensorSSASample,
    values: _CutlassTensorSample | _CutlassTensorSSASample,
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[ThreadData[Any], ThreadData[Any]]:
    """Select largest-key pairs from CUTLASS register tensors.

    ``keys`` and ``values`` supply matching register payloads. ``group``,
    ``k``, ``valid_items``, ``begin_bit``, and ``end_bit`` follow the qualified
    payload contract. Matching register payloads are adapted without mutation.
    External compiler element types necessarily leave both result item types as
    ``Any``. ``temp_storage`` optionally reuses caller-owned block scratch.
    """

@overload
def topk_min_pairs(
    group: _BlockGroup,
    keys: ThreadData[_CutlassTopKKeyT],
    values: ThreadData[_CutlassTopKValueT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[ThreadData[_CutlassTopKKeyT], ThreadData[_CutlassTopKValueT]]:
    """Select smallest-key CUTLASS ``ThreadData`` pairs without mutation.

    ``group`` is a complete one-dimensional physical block. ``keys`` and
    ``values`` have matching positive item counts and supported provider
    dtypes. Uniform ``k``, ``valid_items``, ``begin_bit``, and ``end_bit``
    follow the qualified keys-only contract. Only the first ``k`` unordered
    pairs are defined, and each value remains attached to its key. Both result
    item types are preserved. ``temp_storage`` optionally reuses caller-owned
    block scratch.
    """

@overload
def topk_min_pairs(
    group: _BlockGroup,
    keys: _CutlassTopKKeyT,
    values: _CutlassTopKValueT,
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[_CutlassTopKKeyT, _CutlassTopKValueT]:
    """Select smallest-key pairs from one scalar pair per block member.

    ``keys`` and ``values`` supply one scalar pair. ``group``, ``k``,
    ``valid_items``, ``begin_bit``, and ``end_bit`` follow the qualified
    keys-only contract. Only the unordered first-``k`` block positions are
    defined. Both scalar result types are preserved and values remain attached
    to their keys. ``temp_storage`` optionally reuses caller-owned block scratch.
    """

@overload
def topk_min_pairs(
    group: _BlockGroup,
    keys: _CutlassTensorSample | _CutlassTensorSSASample,
    values: _CutlassTensorSample | _CutlassTensorSSASample,
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[ThreadData[Any], ThreadData[Any]]:
    """Select smallest-key pairs from CUTLASS register tensors.

    ``keys`` and ``values`` supply matching register payloads. ``group``,
    ``k``, ``valid_items``, ``begin_bit``, and ``end_bit`` follow the qualified
    payload contract. Matching register payloads are adapted without mutation.
    External compiler element types necessarily leave both result item types as
    ``Any``. ``temp_storage`` optionally reuses caller-owned block scratch.
    """

__all__ = [
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]
