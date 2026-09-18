# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Out-of-place block TopK operations."""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _group_primitive_marker,
    _portable_group_operation,
)
from ._payload import _validate_common_numeric_value


@_portable_group_operation("topk_min_keys", group_kinds=("block",))
def topk_min_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    k: Any,
    valid_items: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return the minimum keys in an unsorted blocked prefix.

    Inputs are preserved. Only the first ``min(k, valid_items)`` positions
    are defined; omitted ``valid_items`` means the full tile. Ties have
    unspecified order and selection. Every block member participates with
    uniform integer controls in ``[0, block_threads * items_per_thread]``.
    Invalid runtime controls trap before narrowing to CUB's integer ABI.
    """
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "topk_min_keys",
            "keys",
            keys,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
    return _group_primitive_marker(
        "topk_min_keys",
        group,
        keys,
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


@_portable_group_operation("topk_min_pairs", group_kinds=("block",))
def topk_min_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    k: Any,
    valid_items: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return the minimum pairs in an unsorted blocked prefix.

    Inputs are preserved. Only the first ``min(k, valid_items)`` positions
    are defined; omitted ``valid_items`` means the full tile. Ties have
    unspecified order and selection. Every block member participates with
    uniform integer controls in ``[0, block_threads * items_per_thread]``.
    Invalid runtime controls trap before narrowing to CUB's integer ABI.
    """
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "topk_min_pairs",
            "keys",
            keys,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
        _validate_common_numeric_value(
            "topk_min_pairs",
            "values",
            values,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
    return _group_primitive_marker(
        "topk_min_pairs",
        group,
        keys,
        values,
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


@_portable_group_operation("topk_max_keys", group_kinds=("block",))
def topk_max_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    k: Any,
    valid_items: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return the maximum keys in an unsorted blocked prefix.

    Inputs are preserved. Only the first ``min(k, valid_items)`` positions
    are defined; omitted ``valid_items`` means the full tile. Ties have
    unspecified order and selection. Every block member participates with
    uniform integer controls in ``[0, block_threads * items_per_thread]``.
    Invalid runtime controls trap before narrowing to CUB's integer ABI.
    """
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "topk_max_keys",
            "keys",
            keys,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
    return _group_primitive_marker(
        "topk_max_keys",
        group,
        keys,
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


@_portable_group_operation("topk_max_pairs", group_kinds=("block",))
def topk_max_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    k: Any,
    valid_items: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return the maximum pairs in an unsorted blocked prefix.

    Inputs are preserved. Only the first ``min(k, valid_items)`` positions
    are defined; omitted ``valid_items`` means the full tile. Ties have
    unspecified order and selection. Every block member participates with
    uniform integer controls in ``[0, block_threads * items_per_thread]``.
    Invalid runtime controls trap before narrowing to CUB's integer ABI.
    """
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "topk_max_pairs",
            "keys",
            keys,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
        _validate_common_numeric_value(
            "topk_max_pairs",
            "values",
            values,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
    return _group_primitive_marker(
        "topk_max_pairs",
        group,
        keys,
        values,
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


__all__ = ["topk_min_keys", "topk_min_pairs", "topk_max_keys", "topk_max_pairs"]
