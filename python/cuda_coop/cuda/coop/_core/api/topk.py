# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Out-of-place block TopK operations."""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _common_group_operation,
    _group_primitive_marker,
)
from ._payload import _validate_common_numeric_value


@_common_group_operation("topk_min_keys", group_kinds=("block",))
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
    keys : ThreadData
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

    Call this operation inside a kernel compiled by a registered
    backend. The Numba-CUDA-MLIR and CUTLASS implementations accept
    signed runtime counts up to 64 bits and unsigned counts up to 32 bits.
    They reject invalid static counts during compilation and trap
    on invalid runtime counts.

    Qualified Numba-CUDA-MLIR calls also accept fixed-size local arrays;
    qualified CUTLASS calls accept CuTe register payloads. Both return fresh
    per-thread payloads with the same defined-prefix contract.

    See Also
    --------
    cuda.coop.numba_mlir.topk_min_keys
        Numba-CUDA-MLIR payloads and qualified controls.
    cuda.coop.cutlass.topk_min_keys
        CuTe payloads and qualified controls.
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


@_common_group_operation("topk_min_pairs", group_kinds=("block",))
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
    keys : ThreadData
        Fixed-size per-thread keys in blocked order. Supported dtypes
        are signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        float32, and float64.
    values : ThreadData
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

    Call this operation inside a kernel compiled by a registered
    backend. The Numba-CUDA-MLIR and CUTLASS implementations accept
    signed runtime counts up to 64 bits and unsigned counts up to 32 bits.
    They reject invalid static counts during compilation and trap
    on invalid runtime counts.

    Qualified Numba-CUDA-MLIR calls also accept fixed-size local arrays;
    qualified CUTLASS calls accept CuTe register payloads. Both return fresh
    per-thread payloads with the same defined-prefix contract.

    See Also
    --------
    cuda.coop.numba_mlir.topk_min_pairs
        Numba-CUDA-MLIR payloads and qualified controls.
    cuda.coop.cutlass.topk_min_pairs
        CuTe payloads and qualified controls.
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


@_common_group_operation("topk_max_keys", group_kinds=("block",))
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
    keys : ThreadData
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

    Call this operation inside a kernel compiled by a registered
    backend. The Numba-CUDA-MLIR and CUTLASS implementations accept
    signed runtime counts up to 64 bits and unsigned counts up to 32 bits.
    They reject invalid static counts during compilation and trap
    on invalid runtime counts.

    Qualified Numba-CUDA-MLIR calls also accept fixed-size local arrays;
    qualified CUTLASS calls accept CuTe register payloads. Both return fresh
    per-thread payloads with the same defined-prefix contract.

    See Also
    --------
    cuda.coop.numba_mlir.topk_max_keys
        Numba-CUDA-MLIR payloads and qualified controls.
    cuda.coop.cutlass.topk_max_keys
        CuTe payloads and qualified controls.
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


@_common_group_operation("topk_max_pairs", group_kinds=("block",))
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
    keys : ThreadData
        Fixed-size per-thread keys in blocked order. Supported dtypes
        are signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        float32, and float64.
    values : ThreadData
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

    Call this operation inside a kernel compiled by a registered
    backend. The Numba-CUDA-MLIR and CUTLASS implementations accept
    signed runtime counts up to 64 bits and unsigned counts up to 32 bits.
    They reject invalid static counts during compilation and trap
    on invalid runtime counts.

    Qualified Numba-CUDA-MLIR calls also accept fixed-size local arrays;
    qualified CUTLASS calls accept CuTe register payloads. Both return fresh
    per-thread payloads with the same defined-prefix contract.

    See Also
    --------
    cuda.coop.numba_mlir.topk_max_pairs
        Numba-CUDA-MLIR payloads and qualified controls.
    cuda.coop.cutlass.topk_max_pairs
        CuTe payloads and qualified controls.
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
