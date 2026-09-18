# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Merge Sort entry points."""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _group_primitive_marker,
    _portable_group_operation,
)
from ._payload import (
    TempStorageLike,
    ThreadDataLike,
    _ReadableThreadDataLike,
    _validate_common_numeric_value,
    _validate_common_temp_storage,
)


@_portable_group_operation(
    "merge_sort_keys", group_kinds=("block", "warp", "threads_within_warp")
)
def merge_sort_keys(
    group: ThreadGroup,
    keys: _ReadableThreadDataLike[Any],
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[Any]:
    """Return sorted blocked payloads without modifying the inputs."""

    if not isinstance(descending, bool):
        raise TypeError("descending must be a compile-time bool")
    if (valid_items is None) != (oob_default is None):
        raise ValueError("valid_items and oob_default must be provided together")
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "merge_sort_keys",
            "keys",
            keys,
            require_thread_data=True,
            allow_readonly_thread_data=True,
        )
        if temp_storage is not None:
            _validate_common_temp_storage("merge_sort_keys", temp_storage)
    return _group_primitive_marker(
        "merge_sort_keys",
        group,
        keys,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


@_portable_group_operation(
    "merge_sort_pairs", group_kinds=("block", "warp", "threads_within_warp")
)
def merge_sort_pairs(
    group: ThreadGroup,
    keys: _ReadableThreadDataLike[Any],
    values: _ReadableThreadDataLike[Any],
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadDataLike[Any], ThreadDataLike[Any]]:
    """Return sorted blocked payloads without modifying the inputs."""

    if not isinstance(descending, bool):
        raise TypeError("descending must be a compile-time bool")
    if (valid_items is None) != (oob_default is None):
        raise ValueError("valid_items and oob_default must be provided together")
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "merge_sort_pairs",
            "keys",
            keys,
            require_thread_data=True,
            allow_readonly_thread_data=True,
        )
        _validate_common_numeric_value(
            "merge_sort_pairs",
            "values",
            values,
            require_thread_data=True,
            allow_readonly_thread_data=True,
        )
        if temp_storage is not None:
            _validate_common_temp_storage("merge_sort_pairs", temp_storage)
    return _group_primitive_marker(
        "merge_sort_pairs",
        group,
        keys,
        values,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


__all__ = ["merge_sort_keys", "merge_sort_pairs"]
