# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from .._scope import BLOCK_SCOPE as _SCOPE
from .._scope import merge_block_payload as merge_payload
from .._scope import normalize_scan_op as _normalize_scan_op_impl
from .._scope import validate_no_extra_block_args as validate_no_extra_args
from ._dispatch import dispatch_primitive, register_primitive_impl


def _normalize_scan_op(scan_op: Any) -> str:
    return _normalize_scan_op_impl(scan_op, scope=_SCOPE)


def _validate_inclusive_scan_initial_value(initial_value: Any) -> None:
    if initial_value is None:
        return
    raise ValueError(f"{_SCOPE}.inclusive_scan initial_value is not supported")


def _exclusive_sum_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    block_aggregate: Any = None,
    **kwargs: Any,
) -> Any:
    validate_no_extra_args(
        "exclusive_sum",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value",
    )

    from ... import _group_scan as _group_scan_frontend
    from ..._thread_group import this_block

    return _group_scan_frontend._scan(
        this_block(),
        value,
        mode="exclusive",
        scan_op="sum",
        aggregate_output=block_aggregate,
        source="scoped_block",
        **kwargs,
    )


_exclusive_sum_provider._supports_native_thread_data = True
_exclusive_sum_provider._preserves_launch_metadata = True
_exclusive_sum_provider._uses_planned_temp_storage = True
_exclusive_sum_provider._supports_deferred_temp_storage = True


def _exclusive_scan_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    scan_op: Any = None,
    initial_value: Any = None,
    block_aggregate: Any = None,
    **kwargs: Any,
) -> Any:
    validate_no_extra_args(
        "exclusive_scan",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value",
    )
    op = _normalize_scan_op(scan_op)

    from ... import _group_scan as _group_scan_frontend
    from ..._thread_group import this_block

    return _group_scan_frontend._scan(
        this_block(),
        value,
        mode="exclusive",
        scan_op=op,
        initial_value=initial_value,
        aggregate_output=block_aggregate,
        source="scoped_block",
        **kwargs,
    )


_exclusive_scan_provider._supports_native_thread_data = True
_exclusive_scan_provider._preserves_launch_metadata = True
_exclusive_scan_provider._uses_planned_temp_storage = True
_exclusive_scan_provider._supports_deferred_temp_storage = True


def _inclusive_sum_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    block_aggregate: Any = None,
    **kwargs: Any,
) -> Any:
    validate_no_extra_args(
        "inclusive_sum",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value",
    )

    from ... import _group_scan as _group_scan_frontend
    from ..._thread_group import this_block

    return _group_scan_frontend._scan(
        this_block(),
        value,
        mode="inclusive",
        scan_op="sum",
        aggregate_output=block_aggregate,
        source="scoped_block",
        **kwargs,
    )


_inclusive_sum_provider._supports_native_thread_data = True
_inclusive_sum_provider._preserves_launch_metadata = True
_inclusive_sum_provider._uses_planned_temp_storage = True
_inclusive_sum_provider._supports_deferred_temp_storage = True


def _inclusive_scan_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    scan_op: Any = None,
    initial_value: Any = None,
    block_aggregate: Any = None,
    **kwargs: Any,
) -> Any:
    validate_no_extra_args(
        "inclusive_scan",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value",
    )
    op = _normalize_scan_op(scan_op)
    _validate_inclusive_scan_initial_value(initial_value)

    from ... import _group_scan as _group_scan_frontend
    from ..._thread_group import this_block

    return _group_scan_frontend._scan(
        this_block(),
        value,
        mode="inclusive",
        scan_op=op,
        aggregate_output=block_aggregate,
        source="scoped_block",
        **kwargs,
    )


_inclusive_scan_provider._supports_native_thread_data = True
_inclusive_scan_provider._preserves_launch_metadata = True
_inclusive_scan_provider._uses_planned_temp_storage = True
_inclusive_scan_provider._supports_deferred_temp_storage = True


def exclusive_sum(
    value: Any,
    /,
    *args: Any,
    block_aggregate: Any = None,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Return the block-wide exclusive prefix sum for this CuTe thread.

    A size-less ``TempStorage`` selects inferred caller-owned scratch. Shared
    storage inserts the reuse barrier unless ``auto_sync=False`` is requested.
    Explicitly sized legacy storage remains accepted but uncharged.
    """
    structural_payload = {
        "value": value,
        "args": args,
    }
    if block_aggregate is not None:
        structural_payload["block_aggregate"] = block_aggregate
    if temp_storage is not None:
        structural_payload["temp_storage"] = temp_storage
    payload = merge_payload("exclusive_sum", structural_payload, kwargs)
    return dispatch_primitive("exclusive_sum", kwargs=payload)


def exclusive_scan(
    value: Any,
    /,
    *args: Any,
    scan_op: Any = None,
    initial_value: Any = None,
    block_aggregate: Any = None,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Return the block-wide exclusive prefix for a supported scan operator.

    The default operator is sum. Non-sum operators require an explicit
    ``initial_value`` so the provider can seed the first prefix element. A
    runtime initial value must be uniform across the block. ``block_aggregate``
    contains only the reduction of the input items; the initial value is excluded.

    A size-less ``TempStorage`` selects inferred caller-owned scratch. Shared
    storage inserts the reuse barrier unless ``auto_sync=False`` is requested.
    Explicitly sized legacy storage remains accepted but uncharged.
    """
    structural_payload = {
        "value": value,
        "args": args,
        "scan_op": scan_op,
        "initial_value": initial_value,
    }
    if block_aggregate is not None:
        structural_payload["block_aggregate"] = block_aggregate
    if temp_storage is not None:
        structural_payload["temp_storage"] = temp_storage
    payload = merge_payload("exclusive_scan", structural_payload, kwargs)
    return dispatch_primitive("exclusive_scan", kwargs=payload)


def inclusive_sum(
    value: Any,
    /,
    *args: Any,
    block_aggregate: Any = None,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Return the block-wide inclusive prefix sum for this CuTe thread.

    A size-less ``TempStorage`` selects inferred caller-owned scratch. Shared
    storage inserts the reuse barrier unless ``auto_sync=False`` is requested.
    Explicitly sized legacy storage remains accepted but uncharged.
    """
    structural_payload = {
        "value": value,
        "args": args,
    }
    if block_aggregate is not None:
        structural_payload["block_aggregate"] = block_aggregate
    if temp_storage is not None:
        structural_payload["temp_storage"] = temp_storage
    payload = merge_payload("inclusive_sum", structural_payload, kwargs)
    return dispatch_primitive("inclusive_sum", kwargs=payload)


def inclusive_scan(
    value: Any,
    /,
    *args: Any,
    scan_op: Any = None,
    initial_value: Any = None,
    block_aggregate: Any = None,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Return the block-wide inclusive prefix for a supported scan operator.

    Inclusive scans use the CUB operator identity and do not accept
    ``initial_value`` in the current CuTe provider.

    A size-less ``TempStorage`` selects inferred caller-owned scratch. Shared
    storage inserts the reuse barrier unless ``auto_sync=False`` is requested.
    Explicitly sized legacy storage remains accepted but uncharged.
    """
    structural_payload = {
        "value": value,
        "args": args,
        "scan_op": scan_op,
        "initial_value": initial_value,
    }
    if block_aggregate is not None:
        structural_payload["block_aggregate"] = block_aggregate
    if temp_storage is not None:
        structural_payload["temp_storage"] = temp_storage
    payload = merge_payload("inclusive_scan", structural_payload, kwargs)
    return dispatch_primitive("inclusive_scan", kwargs=payload)


def scan(
    value: Any,
    /,
    *args: Any,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    block_aggregate: Any = None,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Dispatch to :func:`exclusive_scan` or :func:`inclusive_scan` by ``mode``.

    A runtime ``initial_value`` must be uniform across the block, and it is not
    included in ``block_aggregate``. A size-less ``TempStorage`` selects
    inferred caller-owned scratch; shared storage inserts the reuse barrier
    unless ``auto_sync=False`` is requested. Explicitly sized legacy storage is
    accepted but uncharged.
    """
    if mode == "exclusive":
        return exclusive_scan(
            value,
            *args,
            scan_op=scan_op,
            initial_value=initial_value,
            block_aggregate=block_aggregate,
            temp_storage=temp_storage,
            **kwargs,
        )
    if mode == "inclusive":
        return inclusive_scan(
            value,
            *args,
            scan_op=scan_op,
            initial_value=initial_value,
            block_aggregate=block_aggregate,
            temp_storage=temp_storage,
            **kwargs,
        )
    raise ValueError(f"{_SCOPE}.scan mode must be 'exclusive' or 'inclusive'")


register_primitive_impl("exclusive_sum", impl=_exclusive_sum_provider)
register_primitive_impl("exclusive_scan", impl=_exclusive_scan_provider)
register_primitive_impl("inclusive_sum", impl=_inclusive_sum_provider)
register_primitive_impl("inclusive_scan", impl=_inclusive_scan_provider)
