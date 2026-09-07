# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable TopK entry points for key and pair variants.

The four direction/result forms share one portable controls validator before
delegation. Qualified backend algorithms, provider rendering, and compiler
lifecycle remain intentionally outside this root family module.
"""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _group_primitive_marker,
    _validate_portable_operation_group,
)
from ._payload import _validate_common_topk_controls


def topk_max_keys(
    group: ThreadGroup,
    keys: Any,
    k: Any,
    /,
    *,
    valid_items: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
) -> Any:
    """Select largest keys through the compiler-selected backend.

    Only the first ``k`` positions are defined. ``temp_storage`` supplies
    optional reusable scratch. Use the qualified ``cuda.coop.<backend>`` API
    for backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_portable_operation_group("topk_max_keys", group)
        _validate_common_topk_controls(
            "topk_max_keys",
            group=group,
            keys=keys,
            k=k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )

    return _group_primitive_marker(
        "topk_max_keys",
        group,
        keys,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
    )


def topk_min_keys(
    group: ThreadGroup,
    keys: Any,
    k: Any,
    /,
    *,
    valid_items: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
) -> Any:
    """Select smallest keys through the compiler-selected backend.

    Only the first ``k`` positions are defined. ``temp_storage`` supplies
    optional reusable scratch. Use the qualified ``cuda.coop.<backend>`` API
    for backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_portable_operation_group("topk_min_keys", group)
        _validate_common_topk_controls(
            "topk_min_keys",
            group=group,
            keys=keys,
            k=k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )

    return _group_primitive_marker(
        "topk_min_keys",
        group,
        keys,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
    )


def topk_max_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    k: Any,
    /,
    *,
    valid_items: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
) -> tuple[Any, Any]:
    """Select largest-key pairs through the compiler-selected backend.

    Only the first ``k`` flattened blocked pairs are defined. That prefix is
    unordered, each value remains attached to its key, and neither input is
    mutated. Use the qualified ``cuda.coop.<backend>`` API for backend-specific
    payloads. ``temp_storage`` supplies optional reusable scratch.
    """

    if _backend_module_name() is not None:
        _validate_portable_operation_group("topk_max_pairs", group)
        _validate_common_topk_controls(
            "topk_max_pairs",
            group=group,
            keys=keys,
            values=values,
            k=k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )

    return _group_primitive_marker(
        "topk_max_pairs",
        group,
        keys,
        values,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
    )


def topk_min_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    k: Any,
    /,
    *,
    valid_items: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
) -> tuple[Any, Any]:
    """Select smallest-key pairs through the compiler-selected backend.

    Only the first ``k`` flattened blocked pairs are defined. That prefix is
    unordered, each value remains attached to its key, and neither input is
    mutated. Use the qualified ``cuda.coop.<backend>`` API for backend-specific
    payloads. ``temp_storage`` supplies optional reusable scratch.
    """

    if _backend_module_name() is not None:
        _validate_portable_operation_group("topk_min_pairs", group)
        _validate_common_topk_controls(
            "topk_min_pairs",
            group=group,
            keys=keys,
            values=values,
            k=k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )

    return _group_primitive_marker(
        "topk_min_pairs",
        group,
        keys,
        values,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
    )


__all__ = [
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]
