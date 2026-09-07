# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first TopK markers for Numba-CUDA-MLIR.

The markers expose fresh key/pair results.  Static bounds and temporary
storage are validated by the TopK lowering path.
"""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


def _topk_keys(operation: str, group: ThreadGroup, keys: Any, k: Any, **kwargs: Any):
    return group_primitive_marker(operation, group, keys, k, **kwargs)


def _topk_pairs(
    operation: str,
    group: ThreadGroup,
    keys: Any,
    values: Any,
    k: Any,
    **kwargs: Any,
):
    return group_primitive_marker(operation, group, keys, values, k, **kwargs)


@group_operation("topk_max_keys")
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
    """Select the largest keys into a fresh fixed-size payload."""

    return _topk_keys(
        "topk_max_keys",
        group,
        keys,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
    )


@group_operation("topk_min_keys")
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
    """Select the smallest keys into a fresh fixed-size payload."""

    return _topk_keys(
        "topk_min_keys",
        group,
        keys,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
    )


@group_operation("topk_max_pairs")
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
    """Select largest-key pairs into fresh fixed-size payloads."""

    return _topk_pairs(
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


@group_operation("topk_min_pairs")
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
    """Select smallest-key pairs into fresh fixed-size payloads."""

    return _topk_pairs(
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
