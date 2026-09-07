# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first merge-sort entrypoints."""

from __future__ import annotations

import operator
from typing import Any

from cuda.coop._core import (
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupMergeSortSemantics,
    LaunchFacts,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block import make_block_merge_sort_semantics

from ._internal._thread_data import _coerce_thread_payload
from ._thread_group import ThreadGroup, _resolve_collective_group_from_launch

_SCOPE = __name__.rsplit(".", 1)[0]

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


def _normalize_compare_op(compare_op: Any, descending: bool | None) -> bool:
    if descending is not None and not isinstance(descending, bool):
        raise TypeError("descending must be a static bool or None")
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
        f"{_SCOPE}.merge_sort supports only the built-in ascending/less and "
        "descending/greater comparison operators"
    )


def _compare_operator(descending: bool) -> CxxOperator:
    name = "greater" if descending else "less"
    return CxxOperator(
        f"::cuda::std::{name}<KeyT>",
        Dependency("KeyT"),
        name="compare_op",
    )


def _make_group_merge_sort_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    key_dtype: Any,
    value_dtype: Any | None,
    items_per_thread: int,
    descending: bool,
    valid_items: Any = None,
    oob_default: Any = None,
    source: str = "cutlass_root",
) -> GroupLoweringPlan:
    """Build the canonical shared-core plan for block or warp MergeSort."""

    primitive = make_block_merge_sort_semantics(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        items_per_thread=items_per_thread,
        compare_operator=_compare_operator(descending),
        valid_items=valid_items,
        oob_default=oob_default,
    )
    call = make_group_primitive_call(
        group,
        GroupMergeSortSemantics(primitive),
        source=source,
    )
    return plan_group_primitive(call, launch)


def _merge_sort(
    group: ThreadGroup,
    keys: Any,
    values: Any | None = None,
    /,
    *args: Any,
    compare_op: Any = None,
    descending: bool | None = None,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: Any = None,
    source: str = "cutlass_root",
    **kwargs: Any,
) -> Any:
    """Internal entrypoint shared by root and scoped compatibility APIs."""

    from ._dsl._launch import infer_launch_facts, pop_launch_metadata
    from ._dsl._scope import merge_payload, validate_no_extra_args

    payload = merge_payload(
        _SCOPE,
        "merge_sort",
        {
            "group": group,
            "keys": keys,
            "values": values,
            "args": args,
            "compare_op": compare_op,
            "descending": descending,
            "valid_items": valid_items,
            "oob_default": oob_default,
        },
        kwargs,
    )
    launch_kwargs = pop_launch_metadata(kwargs)
    validate_no_extra_args(
        _SCOPE,
        "merge_sort",
        args=payload.pop("args"),
        kwargs=kwargs,
        expected=(
            "expects an explicit ThreadGroup and keys, with optional values, "
            "ordering, and block-or-warp partial-tile arguments"
        ),
    )
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.merge_sort group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(f"{_SCOPE}.merge_sort supports block and warp groups")
    primitive_name = "merge_sort_pairs" if values is not None else "merge_sort_keys"
    keys = _coerce_thread_payload(
        keys,
        scope=_SCOPE,
        primitive_name=primitive_name,
        arg_name="keys",
        common_root_payload_kind="thread_data",
    )
    if values is not None:
        values = _coerce_thread_payload(
            values,
            scope=_SCOPE,
            primitive_name=primitive_name,
            arg_name="values",
            common_root_payload_kind="thread_data",
        )
    if (valid_items is None) != (oob_default is None):
        raise ValueError(
            f"{_SCOPE}.merge_sort valid_items and oob_default must be provided together"
        )
    descending = _normalize_compare_op(compare_op, descending)
    launch = infer_launch_facts(
        launch_kwargs,
        scope=_SCOPE,
        primitive_name="merge_sort",
    )
    validated_group = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="merge_sort",
    )

    from ._dsl import _cub_merge_sort_provider as _provider

    return _provider.provider_merge_sort(
        group=validated_group,
        launch=launch,
        keys=keys,
        values=values,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        source=source,
        temp_storage=temp_storage,
    )


def merge_sort_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: Any = None,
    compare_op: Any = None,
) -> Any:
    """Merge-sort keys across an explicit block, physical warp, or logical warp.

    Block groups accept scalar or fixed-size register keys. Physical and logical
    warp groups require fixed-size register keys. A register payload may be
    ``ThreadData``, an rmem tensor, or ``TensorSSA``. All group kinds accept the
    paired ``valid_items``/``oob_default`` partial-tile arguments. Equal keys
    have no stable relative-order guarantee; partial-tile outputs beyond the
    valid sorted prefix are unspecified.
    """

    return _merge_sort(
        group,
        keys,
        None,
        compare_op=compare_op,
        descending=None if compare_op is not None and not descending else descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
        source="cutlass_root",
    )


def merge_sort_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: Any = None,
    compare_op: Any = None,
) -> tuple[Any, Any]:
    """Merge-sort associated key/value pairs across an explicit group.

    One public CUB ``Sort`` invocation updates both arrays. Equal-key ordering
    is unspecified; key/value association is preserved in the valid sorted
    prefix, and partial-tile outputs beyond that prefix are unspecified.
    """

    return _merge_sort(
        group,
        keys,
        values,
        compare_op=compare_op,
        descending=None if compare_op is not None and not descending else descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
        source="cutlass_root",
    )


__all__ = ["merge_sort_keys", "merge_sort_pairs"]
