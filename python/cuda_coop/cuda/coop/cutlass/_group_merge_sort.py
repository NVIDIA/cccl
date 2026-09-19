# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Input-preserving built-in Merge Sort for CuTe block and warp groups."""

from cuda.coop._core.api._payload import (
    _validate_common_temp_storage,
)
from cuda.coop._core.thread_group import ThreadGroup

from ._thread_data import _snapshot_readable_payload
from ._thread_group import (
    _require_complete_warp_partition,
    _resolve_primitive_group_from_launch,
)

_SCOPE = "cuda.coop.cutlass"


def _merge_sort(
    group, keys, values, *, descending, valid_items, oob_default, temp_storage
):
    primitive = "merge_sort_keys" if values is None else "merge_sort_pairs"
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.{primitive} group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.{primitive} requires a block or warp group"
        )
    if not isinstance(descending, bool):
        raise TypeError(f"{_SCOPE}.{primitive} descending must be a compile-time bool")
    if (valid_items is None) != (oob_default is None):
        raise ValueError(
            "Merge Sort valid_items and oob_default must be provided together"
        )
    if temp_storage is not None:
        if group.kind != "block":
            raise ValueError("Merge Sort temp_storage applies only to block groups")
        _validate_common_temp_storage(primitive, temp_storage)
    keys = _snapshot_readable_payload(keys, name="keys", primitive=primitive)
    values = (
        None
        if values is None
        else _snapshot_readable_payload(values, name="values", primitive=primitive)
    )
    if values is not None and keys.items_per_thread != values.items_per_thread:
        raise ValueError(
            "Merge Sort keys and values must have matching items_per_thread extents"
        )

    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._merge_sort import provider_merge_sort

    launch = current_kernel_launch_facts()
    group = _resolve_primitive_group_from_launch(group, launch, feature=primitive)
    _require_complete_warp_partition(
        group, feature=primitive, exact_block_dim=launch.exact_block_dim
    )
    return provider_merge_sort(
        group=group,
        launch=launch,
        keys=keys,
        values=values,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


def merge_sort_keys(
    group,
    keys,
    /,
    *,
    descending=False,
    valid_items=None,
    oob_default=None,
    temp_storage=None,
):
    """Return freshly sorted keys in blocked order, preserving readable inputs.

    Blocks require a power-of-two thread count. Physical and logical warp
    groups require complete enclosing physical warps. All members participate
    with uniform controls. Only the valid prefix of a partial tile is defined;
    its oob_default must sort after valid keys. Equal keys have no stability
    guarantee. Custom comparison predicates are not supported.
    """
    return _merge_sort(
        group,
        keys,
        None,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


def merge_sort_pairs(
    group,
    keys,
    values,
    /,
    *,
    descending=False,
    valid_items=None,
    oob_default=None,
    temp_storage=None,
):
    """Return fresh sorted keys and associated values, preserving both inputs.

    Keys and values have equal per-thread extents and independent numeric
    dtypes. Group, prefix, storage, and ordering rules match merge_sort_keys.
    """
    return _merge_sort(
        group,
        keys,
        values,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


__all__ = ["merge_sort_keys", "merge_sort_pairs"]
