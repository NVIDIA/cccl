# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS merge sort."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, overload

from typing_extensions import TypeVar

from .._typing import _ValidItems as _ValidItems
from ._temp_storage import TempStorage
from ._thread_data import ThreadData
from ._thread_data import _CutlassTensorSample as _CutlassTensorSample
from ._thread_data import _CutlassTensorSSASample as _CutlassTensorSSASample
from ._thread_group import _BlockGroup as _BlockGroup
from ._thread_group import _MergeSortWarpGroup as _MergeSortWarpGroup
from ._typing import _CutlassOrderedItem as _CutlassOrderedItem
from ._typing import _CutlassPairValueT as _CutlassPairValueT

_BlockMergeSortKeyT = TypeVar("_BlockMergeSortKeyT", bound=_CutlassOrderedItem)
_WarpMergeSortKeyT = TypeVar("_WarpMergeSortKeyT", bound=_CutlassOrderedItem)
_CompareOperator: TypeAlias = Literal[
    "<",
    "lt",
    "less",
    "ascending",
    "asc",
    ">",
    "gt",
    "greater",
    "descending",
    "desc",
]

@overload
def merge_sort_keys(
    group: _BlockGroup,
    keys: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorage | None = None,
    compare_op: _CompareOperator | None = None,
) -> ThreadData[Any]:
    """Sort one numeric CUTLASS block register tensor into ``ThreadData``.

    CUTLASS compiler tensor classes do not expose their element type to Python
    type checkers. The compiler validates the block key against the i32, u32,
    i64, u64, f32, and f64 provider set. The qualified adapter returns flattened
    ``ThreadData`` whose item count matches the input register tensor.
    """

@overload
def merge_sort_keys(
    group: _BlockGroup,
    keys: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _CutlassOrderedItem,
    temp_storage: TempStorage | None = None,
    compare_op: _CompareOperator | None = None,
) -> ThreadData[Any]:
    """Partially sort a numeric block register tensor into ``ThreadData``.

    ``valid_items`` and ``oob_default`` are required together. The external
    tensor element type is unavailable to static analysis, so the compiler
    performs the final sentinel compatibility check. The block thread count
    must be a power of two, and the sentinel must sort after every valid key:
    greater for the built-in ascending order and less for descending.
    """

@overload
def merge_sort_keys(
    group: _MergeSortWarpGroup,
    keys: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: None = None,
    compare_op: _CompareOperator | None = None,
) -> ThreadData[Any]:
    """Sort one CUTLASS physical- or logical-warp register tensor.

    The compiler validates the opaque element type against the current warp
    provider set: u8, i32, u32, i64, u64, f32, or f64. The result is flattened
    ``ThreadData`` with the same per-thread item count.
    """

@overload
def merge_sort_keys(
    group: _MergeSortWarpGroup,
    keys: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _CutlassOrderedItem,
    temp_storage: None = None,
    compare_op: _CompareOperator | None = None,
) -> ThreadData[Any]:
    """Partially sort one CUTLASS physical- or logical-warp register tensor.

    The compiler performs the final compatibility check between the opaque
    tensor element type and ``oob_default``. The sentinel must sort after every
    valid key: greater for the built-in ascending order and less for
    descending.
    """

@overload
def merge_sort_keys(
    group: _BlockGroup,
    keys: ThreadData[_BlockMergeSortKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorage | None = None,
    compare_op: _CompareOperator | None = None,
) -> ThreadData[_BlockMergeSortKeyT]:
    """Return fully merge-sorted CUTLASS ``ThreadData`` keys.

    Complete blocks with a power-of-two thread count are supported. The result
    preserves the payload item type and item count.
    ``compare_op`` accepts only the documented built-in less/greater aliases.
    """

@overload
def merge_sort_keys(
    group: _BlockGroup,
    keys: ThreadData[_BlockMergeSortKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _BlockMergeSortKeyT,
    temp_storage: TempStorage | None = None,
    compare_op: _CompareOperator | None = None,
) -> ThreadData[_BlockMergeSortKeyT]:
    """Return partial-tile merge-sorted CUTLASS ``ThreadData`` keys.

    ``valid_items`` and the key-typed ``oob_default`` are required together.
    The block thread count must be a power of two, and the sentinel must sort
    after every valid key: greater for the built-in ascending order and less
    for descending. Only the valid sorted prefix is defined; the payload shape
    is preserved.
    """

@overload
def merge_sort_keys(
    group: _MergeSortWarpGroup,
    keys: ThreadData[_WarpMergeSortKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: None = None,
    compare_op: _CompareOperator | None = None,
) -> ThreadData[_WarpMergeSortKeyT]:
    """Return fully merge-sorted CUTLASS physical- or logical-warp keys.

    The qualified warp provider additionally supports u8, f32, and f64 keys.
    The result preserves the ``ThreadData`` item type and item count.
    """

@overload
def merge_sort_keys(
    group: _MergeSortWarpGroup,
    keys: ThreadData[_WarpMergeSortKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _WarpMergeSortKeyT,
    temp_storage: None = None,
    compare_op: _CompareOperator | None = None,
) -> ThreadData[_WarpMergeSortKeyT]:
    """Return partial-tile sorted CUTLASS physical- or logical-warp keys.

    ``oob_default`` must sort after every valid key: greater for the built-in
    ascending order and less for descending. Only the valid sorted prefix is
    defined.
    """

@overload
def merge_sort_keys(
    group: _BlockGroup,
    keys: _BlockMergeSortKeyT,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorage | None = None,
    compare_op: _CompareOperator | None = None,
) -> _BlockMergeSortKeyT:
    """Return one fully sorted numeric CUTLASS block key per thread."""

@overload
def merge_sort_keys(
    group: _BlockGroup,
    keys: _BlockMergeSortKeyT,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _BlockMergeSortKeyT,
    temp_storage: TempStorage | None = None,
    compare_op: _CompareOperator | None = None,
) -> _BlockMergeSortKeyT:
    """Return one partial-tile sorted numeric CUTLASS block key per thread.

    The block thread count must be a power of two. ``oob_default`` must sort
    after every valid key: greater for the built-in ascending order and less
    for descending. Only the valid sorted prefix is defined.
    """

@overload
def merge_sort_pairs(
    group: _BlockGroup,
    keys: ThreadData[_BlockMergeSortKeyT],
    values: ThreadData[_CutlassPairValueT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorage | None = None,
    compare_op: _CompareOperator | None = None,
) -> tuple[ThreadData[_BlockMergeSortKeyT], ThreadData[_CutlassPairValueT]]:
    """Return fully merge-sorted CUTLASS block key/value payloads."""

@overload
def merge_sort_pairs(
    group: _BlockGroup,
    keys: ThreadData[_BlockMergeSortKeyT],
    values: ThreadData[_CutlassPairValueT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _BlockMergeSortKeyT,
    temp_storage: TempStorage | None = None,
    compare_op: _CompareOperator | None = None,
) -> tuple[ThreadData[_BlockMergeSortKeyT], ThreadData[_CutlassPairValueT]]:
    """Return partial-tile sorted CUTLASS block key/value payloads.

    For a partial tile, provide ``valid_items`` and ``oob_default`` together;
    the sentinel must sort after every valid key under the selected comparator.
    Block thread counts must be powers of two.
    """

@overload
def merge_sort_pairs(
    group: _MergeSortWarpGroup,
    keys: ThreadData[_WarpMergeSortKeyT],
    values: ThreadData[_CutlassPairValueT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: None = None,
    compare_op: _CompareOperator | None = None,
) -> tuple[ThreadData[_WarpMergeSortKeyT], ThreadData[_CutlassPairValueT]]:
    """Return fully merge-sorted CUTLASS warp key/value payloads."""

@overload
def merge_sort_pairs(
    group: _MergeSortWarpGroup,
    keys: ThreadData[_WarpMergeSortKeyT],
    values: ThreadData[_CutlassPairValueT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _WarpMergeSortKeyT,
    temp_storage: None = None,
    compare_op: _CompareOperator | None = None,
) -> tuple[ThreadData[_WarpMergeSortKeyT], ThreadData[_CutlassPairValueT]]:
    """Return partial-tile sorted CUTLASS warp key/value payloads."""

@overload
def merge_sort_pairs(
    group: _BlockGroup,
    keys: _BlockMergeSortKeyT,
    values: _CutlassPairValueT,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorage | None = None,
    compare_op: _CompareOperator | None = None,
) -> tuple[_BlockMergeSortKeyT, _CutlassPairValueT]:
    """Return one fully merge-sorted CUTLASS block key/value pair per thread."""

@overload
def merge_sort_pairs(
    group: _BlockGroup,
    keys: _BlockMergeSortKeyT,
    values: _CutlassPairValueT,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _BlockMergeSortKeyT,
    temp_storage: TempStorage | None = None,
    compare_op: _CompareOperator | None = None,
) -> tuple[_BlockMergeSortKeyT, _CutlassPairValueT]:
    """Return one partial-tile sorted CUTLASS block pair per thread."""

@overload
def merge_sort_pairs(
    group: _BlockGroup,
    keys: _CutlassTensorSample | _CutlassTensorSSASample,
    values: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorage | None = None,
    compare_op: _CompareOperator | None = None,
) -> tuple[ThreadData[Any], ThreadData[Any]]:
    """Return fully sorted block register tensors as CUTLASS payloads."""

@overload
def merge_sort_pairs(
    group: _BlockGroup,
    keys: _CutlassTensorSample | _CutlassTensorSSASample,
    values: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _CutlassOrderedItem,
    temp_storage: TempStorage | None = None,
    compare_op: _CompareOperator | None = None,
) -> tuple[ThreadData[Any], ThreadData[Any]]:
    """Return partial-tile sorted block register tensors as CUTLASS payloads."""

@overload
def merge_sort_pairs(
    group: _MergeSortWarpGroup,
    keys: _CutlassTensorSample | _CutlassTensorSSASample,
    values: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: None = None,
    compare_op: _CompareOperator | None = None,
) -> tuple[ThreadData[Any], ThreadData[Any]]:
    """Return fully sorted warp register tensors as CUTLASS payloads."""

@overload
def merge_sort_pairs(
    group: _MergeSortWarpGroup,
    keys: _CutlassTensorSample | _CutlassTensorSSASample,
    values: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _CutlassOrderedItem,
    temp_storage: None = None,
    compare_op: _CompareOperator | None = None,
) -> tuple[ThreadData[Any], ThreadData[Any]]:
    """Return partial-tile sorted warp register tensors as CUTLASS payloads."""

__all__ = [
    "merge_sort_keys",
    "merge_sort_pairs",
]
