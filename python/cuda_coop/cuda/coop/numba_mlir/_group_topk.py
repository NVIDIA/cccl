# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Out-of-place block TopK operations."""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation(
    "topk_min_keys", family_module="cuda.coop.numba_mlir._compiler._group_topk"
)
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
    return group_primitive_marker(
        "topk_min_keys",
        group,
        keys,
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


@group_operation(
    "topk_min_pairs", family_module="cuda.coop.numba_mlir._compiler._group_topk"
)
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
    return group_primitive_marker(
        "topk_min_pairs",
        group,
        keys,
        values,
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


@group_operation(
    "topk_max_keys", family_module="cuda.coop.numba_mlir._compiler._group_topk"
)
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
    return group_primitive_marker(
        "topk_max_keys",
        group,
        keys,
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


@group_operation(
    "topk_max_pairs", family_module="cuda.coop.numba_mlir._compiler._group_topk"
)
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
    return group_primitive_marker(
        "topk_max_pairs",
        group,
        keys,
        values,
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


__all__ = ["topk_min_keys", "topk_min_pairs", "topk_max_keys", "topk_max_pairs"]
