# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Input-preserving TopK selection for complete one-dimensional CuTe blocks."""

from cuda.coop._core.thread_group import ThreadGroup

from ._temp_storage import TempStorage
from ._thread_data import _snapshot_readable_payload


def _topk(group, keys, values, *, selection, k, valid_items, temp_storage):
    primitive = f"topk_{selection}_{'keys' if values is None else 'pairs'}"
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"cuda.coop.cutlass.{primitive} group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError("TopK supports only complete this_block() groups")
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError("TopK temp_storage must be CUTLASS TempStorage")
    keys = _snapshot_readable_payload(keys, name="keys", primitive=primitive)
    if values is not None:
        values = _snapshot_readable_payload(values, name="values", primitive=primitive)
        if keys.items_per_thread != values.items_per_thread:
            raise ValueError("TopK keys and values must have matching items_per_thread")

    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._topk import provider_topk

    return provider_topk(
        group=group,
        launch=current_kernel_launch_facts(),
        keys=keys,
        values=values,
        selection=selection,
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def topk_min_keys(group, keys, /, *, k, valid_items=None, temp_storage=None):
    """Return fresh payloads containing the block's smallest keys.

    All threads in a one-dimensional block participate with uniform k and
    valid_items counts in [0, block_size * items_per_thread]. Omitted valid_items
    selects the full input tile. Only min(k, valid_items) blocked positions are
    defined. Selected keys are unsorted; ties have no stability guarantee.
    Inputs and selected floating-point key bits are preserved. Qualified CuTe
    register payloads are converted to ThreadData. NaNs have no numeric ordering
    guarantee. Scratch is automatic unless temp_storage is provided.
    """
    return _topk(
        group,
        keys,
        None,
        selection="min",
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def topk_max_keys(group, keys, /, *, k, valid_items=None, temp_storage=None):
    """Return the block's largest keys under the topk_min_keys contract."""
    return _topk(
        group,
        keys,
        None,
        selection="max",
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def topk_min_pairs(group, keys, values, /, *, k, valid_items=None, temp_storage=None):
    """Select the smallest keys and their values into fresh payloads.

    Keys and values have matching per-thread extents and independently chosen
    numeric dtypes. Both inputs are preserved. Defined output positions, count
    bounds, participation, and storage rules follow topk_min_keys.
    """
    if values is None:
        raise TypeError("TopK values must be a numeric ThreadData payload")
    return _topk(
        group,
        keys,
        values,
        selection="min",
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def topk_max_pairs(group, keys, values, /, *, k, valid_items=None, temp_storage=None):
    """Select the largest key/value pairs under the topk_min_pairs contract."""
    if values is None:
        raise TypeError("TopK values must be a numeric ThreadData payload")
    return _topk(
        group,
        keys,
        values,
        selection="max",
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


__all__ = ["topk_min_keys", "topk_min_pairs", "topk_max_keys", "topk_max_pairs"]
