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
    """Sort keys across a block or warp without changing the input.

    This qualified form of :func:`cuda.coop.merge_sort_keys` also accepts
    CuTe register payloads. Each thread contributes the same fixed number of
    items in blocked order, and all members of the group must participate.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, ``this_warp()``, or
        ``this_warp().group_by(width)``. Blocks require a power-of-two total
        thread count and may be multidimensional. Logical warp widths are
        1, 2, 4, 8, 16, and 32; their enclosing physical warps must be complete.
    keys : ThreadData or CuTe register payload
        Per-thread keys in blocked order, with a positive, compile-time
        extent. Read-only payloads are accepted. CuTe register-memory tensors
        and ``TensorSSA`` values are converted through
        :meth:`ThreadData.from_payload <cuda.coop.cutlass.ThreadData.from_payload>`.
        Supported element types are signed and unsigned 8-, 16-, 32-, and
        64-bit integers, ``Float32``, and ``Float64``.
    descending : bool, optional
        Compile-time order selector. The default is ascending order.
        Custom comparison predicates are not supported.
    valid_items : integer, optional
        Group-uniform count in ``[0, group_size * items_per_thread]``.
        Only this blocked input prefix participates in the sort. Supply
        ``oob_default`` together with this argument for a partial tile.
        Runtime counts accept signed integer types up to 64 bits or unsigned
        integer types up to 32 bits. Invalid static counts fail compilation;
        invalid runtime counts trap before conversion to CUB's count type.
    oob_default : scalar, optional
        Group-uniform sentinel that sorts after valid keys: an upper bound
        for ascending order or a lower bound for descending order. A runtime
        CuTe scalar must match the key type exactly. Representable Python
        numeric literals are converted to the key type.
    temp_storage : TempStorage, optional
        Explicit block scratch. Omit it for automatic allocation. Warp groups
        always use automatic storage with independent slices per group.
        If the descriptor has ``auto_sync=False``, synchronize the block
        before reusing its storage.

    Returns
    -------
    ThreadData
        Fresh sorted keys with the input element type and per-thread extent,
        including when the input is a CuTe register payload. Output is blocked:
        thread ``t`` owns sorted positions ``t * items_per_thread + i``.
        Only the first ``valid_items`` positions are defined for a partial
        tile. Equal keys have no stability guarantee.

    See Also
    --------
    cuda.coop.cutlass.merge_sort_pairs
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
    """Sort keys and their associated values without changing either input.

    This qualified form of :func:`cuda.coop.merge_sort_pairs` accepts CuTe
    register payloads for either operand. Group participation and ordering
    follow :func:`cuda.coop.cutlass.merge_sort_keys`.

    Parameters
    ----------
    group : ThreadGroup
        Complete block, physical warp, or supported logical warp. Block
        thread counts must be powers of two; enclosing physical warps must
        be complete for warp operations.
    keys, values : ThreadData or CuTe register payload
        Fixed-size per-thread payloads with equal extents, in blocked order.
        Either operand may be read-only or use a CuTe register representation.
        Key and value types are independent: signed and unsigned 8-, 16-,
        32-, and 64-bit integers, ``Float32``, and ``Float64`` are supported.
    descending : bool, optional
        Compile-time order selector. The default is ascending key order.
        Custom comparison predicates are not supported.
    valid_items : integer, optional
        Group-uniform length of the valid blocked input prefix, from zero
        through the group tile size. Supply it together with ``oob_default``.
        Count types and runtime checks match ``merge_sort_keys``.
    oob_default : scalar, optional
        Key sentinel that sorts after valid keys. Runtime CuTe scalars must
        have the key type; representable Python literals are converted to it.
    temp_storage : TempStorage, optional
        Explicit scratch for block groups only. Warp storage is automatic.
        Retain automatic synchronization or synchronize before scratch reuse.

    Returns
    -------
    tuple[ThreadData, ThreadData]
        Fresh sorted keys and corresponding values, each retaining its own
        input element type and extent. CuTe register inputs also return
        ``ThreadData``. Only the valid blocked prefix is defined for a partial
        tile. Key/value associations are preserved; equal keys have no
        stability guarantee.

    See Also
    --------
    cuda.coop.cutlass.merge_sort_keys
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
