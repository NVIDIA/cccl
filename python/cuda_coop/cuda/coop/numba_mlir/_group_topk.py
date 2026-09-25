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
    """Select the smallest keys in a block.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block, obtained with ``this_block()``.
        Every thread in the block must call this operation.
    keys : ThreadData or local array
        Fixed-size per-thread keys in blocked order. Supported dtypes
        are signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        float32, and float64.
    k : int
        Requested number of selected items. May be static or runtime,
        must be uniform across the block, and must lie in ``[0, N]``,
        where ``N = block_threads * items_per_thread``.
    valid_items : int, optional
        Number of valid input items in the blocked tile prefix.
        Defaults to ``N`` and has the same range and uniformity
        requirements as ``k``. If ``k > valid_items``, all valid
        items are selected.
    temp_storage : TempStorage, optional
        Shared scratch descriptor. Omit it to let the compiler allocate
        storage. With ``auto_sync=False``, synchronize the block before
        reusing the descriptor in another primitive.

    Returns
    -------
    selected_keys : per-thread payload
        A new payload with the input dtype and per-thread extent.
        Only the first ``min(k, valid_items)`` blocked tile positions
        are defined. Position ``thread_rank * items_per_thread + i``
        belongs to element ``i`` of that thread. Remaining positions
        must not be read or stored.

    Notes
    -----
    Selection preserves the input payloads. Results are unsorted;
    selection and ordering among equal keys are unspecified.
    Zero ``k`` or ``valid_items`` produces no defined output items.
    Positive and negative floating-point zero compare as equal, and
    selected keys retain their original bits. NaNs have no guaranteed
    numeric ordering.

    Runtime count dtypes are signed integers up to 64 bits or
    unsigned integers up to 32 bits. Invalid static counts fail
    compilation; invalid runtime counts trap before conversion to
    CUB's 32-bit count type.

    This implementation uses the private CUB ``cub::detail::block_topk``
    ``min_keys`` operation in ``cub/block/block_topk.cuh``.
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
    """Select the smallest key/value pairs in a block.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block, obtained with ``this_block()``.
        Every thread in the block must call this operation.
    keys : ThreadData or local array
        Fixed-size per-thread keys in blocked order. Supported dtypes
        are signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        float32, and float64.
    values : ThreadData or local array
        Values paired with ``keys``, with the same per-thread extent.
        The value dtype may differ from the key dtype and must be in
        the same supported numeric profile.
    k : int
        Requested number of selected items. May be static or runtime,
        must be uniform across the block, and must lie in ``[0, N]``,
        where ``N = block_threads * items_per_thread``.
    valid_items : int, optional
        Number of valid input items in the blocked tile prefix.
        Defaults to ``N`` and has the same range and uniformity
        requirements as ``k``. If ``k > valid_items``, all valid
        items are selected.
    temp_storage : TempStorage, optional
        Shared scratch descriptor. Omit it to let the compiler allocate
        storage. With ``auto_sync=False``, synchronize the block before
        reusing the descriptor in another primitive.

    Returns
    -------
    selected_keys, selected_values : tuple of per-thread payloads
        New payloads with the input dtypes and per-thread extent.
        Only the first ``min(k, valid_items)`` blocked tile positions
        are defined. Position ``thread_rank * items_per_thread + i``
        belongs to element ``i`` of that thread. Remaining positions
        must not be read or stored.

    Notes
    -----
    Selection preserves the input payloads. Results are unsorted;
    selection and ordering among equal keys are unspecified.
    Each selected value remains paired with its original key.
    Zero ``k`` or ``valid_items`` produces no defined output items.
    Positive and negative floating-point zero compare as equal, and
    selected keys retain their original bits. NaNs have no guaranteed
    numeric ordering.

    Runtime count dtypes are signed integers up to 64 bits or
    unsigned integers up to 32 bits. Invalid static counts fail
    compilation; invalid runtime counts trap before conversion to
    CUB's 32-bit count type.

    This implementation uses the private CUB ``cub::detail::block_topk``
    ``min_pairs`` operation in ``cub/block/block_topk.cuh``.
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
    """Select the largest keys in a block.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block, obtained with ``this_block()``.
        Every thread in the block must call this operation.
    keys : ThreadData or local array
        Fixed-size per-thread keys in blocked order. Supported dtypes
        are signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        float32, and float64.
    k : int
        Requested number of selected items. May be static or runtime,
        must be uniform across the block, and must lie in ``[0, N]``,
        where ``N = block_threads * items_per_thread``.
    valid_items : int, optional
        Number of valid input items in the blocked tile prefix.
        Defaults to ``N`` and has the same range and uniformity
        requirements as ``k``. If ``k > valid_items``, all valid
        items are selected.
    temp_storage : TempStorage, optional
        Shared scratch descriptor. Omit it to let the compiler allocate
        storage. With ``auto_sync=False``, synchronize the block before
        reusing the descriptor in another primitive.

    Returns
    -------
    selected_keys : per-thread payload
        A new payload with the input dtype and per-thread extent.
        Only the first ``min(k, valid_items)`` blocked tile positions
        are defined. Position ``thread_rank * items_per_thread + i``
        belongs to element ``i`` of that thread. Remaining positions
        must not be read or stored.

    Notes
    -----
    Selection preserves the input payloads. Results are unsorted;
    selection and ordering among equal keys are unspecified.
    Zero ``k`` or ``valid_items`` produces no defined output items.
    Positive and negative floating-point zero compare as equal, and
    selected keys retain their original bits. NaNs have no guaranteed
    numeric ordering.

    Runtime count dtypes are signed integers up to 64 bits or
    unsigned integers up to 32 bits. Invalid static counts fail
    compilation; invalid runtime counts trap before conversion to
    CUB's 32-bit count type.

    This implementation uses the private CUB ``cub::detail::block_topk``
    ``max_keys`` operation in ``cub/block/block_topk.cuh``.
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
    """Select the largest key/value pairs in a block.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block, obtained with ``this_block()``.
        Every thread in the block must call this operation.
    keys : ThreadData or local array
        Fixed-size per-thread keys in blocked order. Supported dtypes
        are signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        float32, and float64.
    values : ThreadData or local array
        Values paired with ``keys``, with the same per-thread extent.
        The value dtype may differ from the key dtype and must be in
        the same supported numeric profile.
    k : int
        Requested number of selected items. May be static or runtime,
        must be uniform across the block, and must lie in ``[0, N]``,
        where ``N = block_threads * items_per_thread``.
    valid_items : int, optional
        Number of valid input items in the blocked tile prefix.
        Defaults to ``N`` and has the same range and uniformity
        requirements as ``k``. If ``k > valid_items``, all valid
        items are selected.
    temp_storage : TempStorage, optional
        Shared scratch descriptor. Omit it to let the compiler allocate
        storage. With ``auto_sync=False``, synchronize the block before
        reusing the descriptor in another primitive.

    Returns
    -------
    selected_keys, selected_values : tuple of per-thread payloads
        New payloads with the input dtypes and per-thread extent.
        Only the first ``min(k, valid_items)`` blocked tile positions
        are defined. Position ``thread_rank * items_per_thread + i``
        belongs to element ``i`` of that thread. Remaining positions
        must not be read or stored.

    Notes
    -----
    Selection preserves the input payloads. Results are unsorted;
    selection and ordering among equal keys are unspecified.
    Each selected value remains paired with its original key.
    Zero ``k`` or ``valid_items`` produces no defined output items.
    Positive and negative floating-point zero compare as equal, and
    selected keys retain their original bits. NaNs have no guaranteed
    numeric ordering.

    Runtime count dtypes are signed integers up to 64 bits or
    unsigned integers up to 32 bits. Invalid static counts fail
    compilation; invalid runtime counts trap before conversion to
    CUB's 32-bit count type.

    This implementation uses the private CUB ``cub::detail::block_topk``
    ``max_pairs`` operation in ``cub/block/block_topk.cuh``.
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
