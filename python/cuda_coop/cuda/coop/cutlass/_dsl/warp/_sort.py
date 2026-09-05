# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import operator
from typing import Any

from .._launch import resolve_threads_in_warp as _resolve_threads_in_warp
from .._scope import WARP_SCOPE as _SCOPE
from .._scope import merge_warp_payload as merge_payload
from .._scope import validate_no_extra_warp_args as validate_no_extra_args
from ._dispatch import dispatch_primitive, register_primitive_impl

_COMPARE_OP_ALIASES = {
    "<": False,
    "lt": False,
    "less": False,
    "ascending": False,
    "asc": False,
    ">": True,
    "gt": True,
    "greater": True,
    "descending": True,
    "desc": True,
}

_CALLABLE_COMPARE_OP_ALIASES = {
    operator.lt: False,
    operator.gt: True,
}

_CALLABLE_COMPARE_OP_NAME_ALIASES = {
    ("_operator", "lt"): False,
    ("_operator", "gt"): True,
    ("operator", "lt"): False,
    ("operator", "gt"): True,
    ("numpy", "less"): False,
    ("numpy", "greater"): True,
}


def _validate_partial_tile_args(
    primitive_name: str,
    *,
    valid_items: Any,
    oob_default: Any,
) -> None:
    if (valid_items is None) != (oob_default is None):
        raise ValueError(
            f"{_SCOPE}.{primitive_name} valid_items and oob_default "
            "must be provided together"
        )


def _normalize_compare_op(compare_op: Any, descending: bool | None) -> bool:
    if descending is not None and not isinstance(descending, bool):
        raise TypeError("descending must be a bool or None")
    if compare_op is None:
        return bool(descending)

    try:
        op_descending = _COMPARE_OP_ALIASES[compare_op]
    except (KeyError, TypeError):
        pass
    else:
        if descending is not None and descending != op_descending:
            raise ValueError("compare_op and descending request conflicting orders")
        return op_descending if descending is None else descending

    try:
        op_descending = _CALLABLE_COMPARE_OP_ALIASES[compare_op]
    except (KeyError, TypeError):
        pass
    else:
        if descending is not None and descending != op_descending:
            raise ValueError("compare_op and descending request conflicting orders")
        return op_descending if descending is None else descending

    if callable(compare_op):
        module = getattr(compare_op, "__module__", "")
        name = getattr(compare_op, "__name__", "")
        try:
            op_descending = _CALLABLE_COMPARE_OP_NAME_ALIASES[(module, name)]
        except KeyError:
            pass
        else:
            if descending is not None and descending != op_descending:
                raise ValueError("compare_op and descending request conflicting orders")
            return op_descending if descending is None else descending

    raise NotImplementedError(
        f"{_SCOPE}.merge_sort currently supports ascending/less "
        "and descending/greater compare operators. Arbitrary Python callables are "
        "not lowered yet."
    )


def _merge_sort_keys_provider(
    *,
    keys: Any,
    args: tuple[Any, ...] = (),
    compare_op: Any = None,
    descending: bool | None = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    oob_default: Any = None,
    **kwargs: Any,
) -> Any:
    if args:
        validate_no_extra_args(
            "merge_sort_keys",
            args=args,
            kwargs={},
            expected=(
                "accepts keys plus optional compare_op, descending, and threads_in_warp"
            ),
        )

    _validate_partial_tile_args(
        "merge_sort_keys",
        valid_items=valid_items,
        oob_default=oob_default,
    )

    from ... import _group_merge_sort as _group_frontend
    from ..._thread_group import this_warp

    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE,
        "merge_sort_keys",
        threads_in_warp,
    )
    group = this_warp()
    if threads_in_warp != 32:
        group = group.group_by(threads_in_warp)
    return _group_frontend._merge_sort(
        group,
        keys,
        None,
        descending=_normalize_compare_op(compare_op, descending),
        valid_items=valid_items,
        oob_default=oob_default,
        source="scoped_warp",
        **kwargs,
    )


_merge_sort_keys_provider._supports_native_thread_data = True
_merge_sort_keys_provider._preserves_launch_metadata = True
_merge_sort_keys_provider._uses_planned_temp_storage = True


def _merge_sort_pairs_provider(
    *,
    keys: Any,
    values: Any,
    args: tuple[Any, ...] = (),
    compare_op: Any = None,
    descending: bool | None = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    oob_default: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    if args:
        validate_no_extra_args(
            "merge_sort_pairs",
            args=args,
            kwargs={},
            expected=(
                "accepts keys, values, plus optional compare_op, descending, "
                "and threads_in_warp"
            ),
        )

    _validate_partial_tile_args(
        "merge_sort_pairs",
        valid_items=valid_items,
        oob_default=oob_default,
    )

    from ... import _group_merge_sort as _group_frontend
    from ..._thread_group import this_warp

    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE,
        "merge_sort_pairs",
        threads_in_warp,
    )
    group = this_warp()
    if threads_in_warp != 32:
        group = group.group_by(threads_in_warp)
    return _group_frontend._merge_sort(
        group,
        keys,
        values,
        descending=_normalize_compare_op(compare_op, descending),
        valid_items=valid_items,
        oob_default=oob_default,
        source="scoped_warp",
        **kwargs,
    )


_merge_sort_pairs_provider._supports_native_thread_data = True
_merge_sort_pairs_provider._preserves_launch_metadata = True
_merge_sort_pairs_provider._uses_planned_temp_storage = True


def merge_sort_keys(
    keys: Any,
    /,
    *args: Any,
    compare_op: Any = None,
    descending: bool | None = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    oob_default: Any = None,
    **kwargs: Any,
) -> Any:
    """Return CUB ``WarpMergeSort`` sorted per-lane keys.

    ``keys`` must be a ``ThreadData`` payload whose items are in blocked order.
    ``threads_in_warp`` selects a power-of-two logical warp size in ``[1, 32]``.
    ``valid_items`` and ``oob_default`` may be provided together for a partial
    tile; output beyond the valid sorted prefix is unspecified.
    ``compare_op`` accepts the built-in less/greater spellings and known
    ``operator``/NumPy aliases; arbitrary Python device callables are not
    lowered by the current CuTe provider.
    """
    _validate_partial_tile_args(
        "merge_sort_keys",
        valid_items=valid_items,
        oob_default=oob_default,
    )
    payload = merge_payload(
        "merge_sort_keys",
        {
            "keys": keys,
            "args": args,
            "compare_op": compare_op,
            "descending": descending,
            "threads_in_warp": threads_in_warp,
            "valid_items": valid_items,
            "oob_default": oob_default,
        },
        kwargs,
    )
    return dispatch_primitive("merge_sort_keys", kwargs=payload)


def merge_sort_pairs(
    keys: Any,
    values: Any,
    /,
    *args: Any,
    compare_op: Any = None,
    descending: bool | None = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    oob_default: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Return CUB ``WarpMergeSort`` sorted key/value ``ThreadData`` pairs.

    Keys and values must both be ``ThreadData`` with matching
    ``items_per_thread``. The output preserves key/value association after
    sorting. Ordering and partial-tile options match :func:`merge_sort_keys`.
    """
    _validate_partial_tile_args(
        "merge_sort_pairs",
        valid_items=valid_items,
        oob_default=oob_default,
    )
    payload = merge_payload(
        "merge_sort_pairs",
        {
            "keys": keys,
            "values": values,
            "args": args,
            "compare_op": compare_op,
            "descending": descending,
            "threads_in_warp": threads_in_warp,
            "valid_items": valid_items,
            "oob_default": oob_default,
        },
        kwargs,
    )
    return dispatch_primitive("merge_sort_pairs", kwargs=payload)


register_primitive_impl("merge_sort_keys", impl=_merge_sort_keys_provider)
register_primitive_impl("merge_sort_pairs", impl=_merge_sort_pairs_provider)
