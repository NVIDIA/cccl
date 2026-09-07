# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from .._launch import resolve_threads_in_warp as _resolve_threads_in_warp
from .._scope import WARP_SCOPE as _SCOPE
from .._scope import merge_warp_payload as merge_payload
from .._scope import normalize_scan_op as _normalize_scan_op_impl
from .._scope import validate_no_extra_warp_args as validate_no_extra_args
from ._dispatch import dispatch_primitive, register_primitive_impl


def _normalize_scan_op(scan_op: Any) -> str:
    return _normalize_scan_op_impl(scan_op, scope=_SCOPE)


def _validate_inclusive_scan_initial_value(initial_value: Any) -> None:
    if initial_value is None:
        return
    raise ValueError(f"{_SCOPE}.inclusive_scan initial_value is not supported")


def _uses_group_scan(
    value: Any,
    *,
    threads_in_warp: int,
    valid_items: Any,
) -> bool:
    from .._thread_data import ThreadData

    return (
        threads_in_warp == 32
        and valid_items is None
        and not isinstance(value, ThreadData)
    )


def _exclusive_sum_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    threads_in_warp: int = 32,
    valid_items: Any = None,
    warp_aggregate: Any = None,
    **kwargs: Any,
) -> Any:
    validate_no_extra_args(
        "exclusive_sum",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value and optional valid_items",
    )
    threads_in_warp = _resolve_threads_in_warp(_SCOPE, "exclusive_sum", threads_in_warp)

    if _uses_group_scan(
        value,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
    ):
        from ... import _group_scan as _group_scan_frontend
        from ..._thread_group import this_warp

        return _group_scan_frontend._scan(
            this_warp(),
            value,
            mode="exclusive",
            scan_op="sum",
            aggregate_output=warp_aggregate,
            **kwargs,
        )

    from . import _provider

    return _provider.provider_exclusive_sum(
        value=value,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        warp_aggregate=warp_aggregate,
    )


def _exclusive_scan_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    scan_op: Any = None,
    initial_value: Any = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    warp_aggregate: Any = None,
    **kwargs: Any,
) -> Any:
    validate_no_extra_args(
        "exclusive_scan",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value and optional valid_items",
    )
    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE, "exclusive_scan", threads_in_warp
    )
    op = _normalize_scan_op(scan_op)

    if _uses_group_scan(
        value,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
    ):
        from ... import _group_scan as _group_scan_frontend
        from ..._thread_group import this_warp

        return _group_scan_frontend._scan(
            this_warp(),
            value,
            mode="exclusive",
            scan_op=op,
            initial_value=initial_value,
            aggregate_output=warp_aggregate,
            **kwargs,
        )

    from . import _provider

    return _provider.provider_exclusive_scan(
        value=value,
        op=op,
        initial_value=initial_value,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        warp_aggregate=warp_aggregate,
    )


def _inclusive_sum_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    threads_in_warp: int = 32,
    valid_items: Any = None,
    warp_aggregate: Any = None,
    **kwargs: Any,
) -> Any:
    validate_no_extra_args(
        "inclusive_sum",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value and optional valid_items",
    )
    threads_in_warp = _resolve_threads_in_warp(_SCOPE, "inclusive_sum", threads_in_warp)

    if _uses_group_scan(
        value,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
    ):
        from ... import _group_scan as _group_scan_frontend
        from ..._thread_group import this_warp

        return _group_scan_frontend._scan(
            this_warp(),
            value,
            mode="inclusive",
            scan_op="sum",
            aggregate_output=warp_aggregate,
            **kwargs,
        )

    from . import _provider

    return _provider.provider_inclusive_sum(
        value=value,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        warp_aggregate=warp_aggregate,
    )


def _inclusive_scan_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    scan_op: Any = None,
    initial_value: Any = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    warp_aggregate: Any = None,
    **kwargs: Any,
) -> Any:
    validate_no_extra_args(
        "inclusive_scan",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value and optional valid_items",
    )
    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE, "inclusive_scan", threads_in_warp
    )
    op = _normalize_scan_op(scan_op)
    _validate_inclusive_scan_initial_value(initial_value)

    if _uses_group_scan(
        value,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
    ):
        from ... import _group_scan as _group_scan_frontend
        from ..._thread_group import this_warp

        return _group_scan_frontend._scan(
            this_warp(),
            value,
            mode="inclusive",
            scan_op=op,
            aggregate_output=warp_aggregate,
            **kwargs,
        )

    from . import _provider

    return _provider.provider_inclusive_scan(
        value=value,
        op=op,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        warp_aggregate=warp_aggregate,
    )


def exclusive_sum(
    value: Any,
    /,
    *args: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    warp_aggregate: Any = None,
    **kwargs: Any,
) -> Any:
    """Return the exclusive CUB ``WarpScan::ExclusiveSum`` prefix for this lane.

    ``warp_aggregate`` may be a one-item ``ThreadData`` output that receives the
    CUB warp aggregate for every participating lane.
    """
    structural_payload = {
        "value": value,
        "args": args,
        "threads_in_warp": threads_in_warp,
    }
    if valid_items is not None:
        structural_payload["valid_items"] = valid_items
    if warp_aggregate is not None:
        structural_payload["warp_aggregate"] = warp_aggregate
    payload = merge_payload("exclusive_sum", structural_payload, kwargs)
    return dispatch_primitive("exclusive_sum", kwargs=payload)


def exclusive_scan(
    value: Any,
    /,
    *args: Any,
    scan_op: Any = None,
    initial_value: Any = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    warp_aggregate: Any = None,
    **kwargs: Any,
) -> Any:
    """Return an exclusive CUB ``WarpScan`` prefix for this logical-warp lane.

    Sum scans default to CUB's identity-zero ``ExclusiveSum`` overload. Other
    built-in operators require ``initial_value`` so lane 0 has an explicit
    seed, matching the block-scoped CuTe API. ``warp_aggregate`` may be a
    one-item ``ThreadData`` output. A runtime initial value must be uniform
    across the warp and is excluded from the warp aggregate.
    """
    structural_payload = {
        "value": value,
        "args": args,
        "scan_op": scan_op,
        "initial_value": initial_value,
        "threads_in_warp": threads_in_warp,
    }
    if valid_items is not None:
        structural_payload["valid_items"] = valid_items
    if warp_aggregate is not None:
        structural_payload["warp_aggregate"] = warp_aggregate
    payload = merge_payload("exclusive_scan", structural_payload, kwargs)
    return dispatch_primitive("exclusive_scan", kwargs=payload)


def inclusive_sum(
    value: Any,
    /,
    *args: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    warp_aggregate: Any = None,
    **kwargs: Any,
) -> Any:
    """Return the inclusive CUB ``WarpScan::InclusiveSum`` prefix for this lane.

    ``warp_aggregate`` may be a one-item ``ThreadData`` output that receives the
    CUB warp aggregate for every participating lane.
    """
    structural_payload = {
        "value": value,
        "args": args,
        "threads_in_warp": threads_in_warp,
    }
    if valid_items is not None:
        structural_payload["valid_items"] = valid_items
    if warp_aggregate is not None:
        structural_payload["warp_aggregate"] = warp_aggregate
    payload = merge_payload("inclusive_sum", structural_payload, kwargs)
    return dispatch_primitive("inclusive_sum", kwargs=payload)


def inclusive_scan(
    value: Any,
    /,
    *args: Any,
    scan_op: Any = None,
    initial_value: Any = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    warp_aggregate: Any = None,
    **kwargs: Any,
) -> Any:
    """Return an inclusive CUB ``WarpScan`` prefix for this logical-warp lane.

    ``warp_aggregate`` may be a one-item ``ThreadData`` output.
    """
    structural_payload = {
        "value": value,
        "args": args,
        "scan_op": scan_op,
        "initial_value": initial_value,
        "threads_in_warp": threads_in_warp,
    }
    if valid_items is not None:
        structural_payload["valid_items"] = valid_items
    if warp_aggregate is not None:
        structural_payload["warp_aggregate"] = warp_aggregate
    payload = merge_payload("inclusive_scan", structural_payload, kwargs)
    return dispatch_primitive("inclusive_scan", kwargs=payload)


def scan(
    value: Any,
    /,
    *args: Any,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    warp_aggregate: Any = None,
    **kwargs: Any,
) -> Any:
    """Dispatch to :func:`exclusive_scan` or :func:`inclusive_scan` by ``mode``.

    A runtime ``initial_value`` must be uniform across the warp and is not
    included in ``warp_aggregate``.
    """
    if mode == "exclusive":
        return exclusive_scan(
            value,
            *args,
            scan_op=scan_op,
            initial_value=initial_value,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
            **kwargs,
        )
    if mode == "inclusive":
        return inclusive_scan(
            value,
            *args,
            scan_op=scan_op,
            initial_value=initial_value,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
            **kwargs,
        )
    raise ValueError(f"{_SCOPE}.scan mode must be 'exclusive' or 'inclusive'")


_exclusive_sum_provider._supports_native_thread_data = True
_exclusive_scan_provider._supports_native_thread_data = True
_inclusive_sum_provider._supports_native_thread_data = True
_inclusive_scan_provider._supports_native_thread_data = True
_exclusive_sum_provider._preserves_launch_metadata = True
_exclusive_scan_provider._preserves_launch_metadata = True
_inclusive_sum_provider._preserves_launch_metadata = True
_inclusive_scan_provider._preserves_launch_metadata = True

register_primitive_impl("exclusive_sum", impl=_exclusive_sum_provider)
register_primitive_impl("exclusive_scan", impl=_exclusive_scan_provider)
register_primitive_impl("inclusive_sum", impl=_inclusive_sum_provider)
register_primitive_impl("inclusive_scan", impl=_inclusive_scan_provider)
