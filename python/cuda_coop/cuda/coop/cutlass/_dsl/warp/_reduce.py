# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from .._launch import resolve_threads_in_warp as _resolve_threads_in_warp
from .._scope import WARP_SCOPE as _SCOPE
from .._scope import merge_warp_payload as merge_payload
from .._scope import normalize_reduce_op as _normalize_reduce_op_impl
from .._scope import validate_no_extra_warp_args as validate_no_extra_args
from ._dispatch import dispatch_primitive, register_primitive_impl


def _normalize_reduce_op(binary_op: Any) -> str:
    return _normalize_reduce_op_impl(binary_op, scope=_SCOPE)


def _sum_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    threads_in_warp: int = 32,
    valid_items: Any = None,
    **kwargs: Any,
) -> Any:
    if args:
        if len(args) != 1 or valid_items is not None:
            raise TypeError(
                f"{_SCOPE}.sum currently expects one positional value and "
                "at most one valid_items"
            )
        valid_items = args[0]
        args = ()
    validate_no_extra_args(
        "sum",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value and optional valid_items",
    )
    threads_in_warp = _resolve_threads_in_warp(_SCOPE, "sum", threads_in_warp)

    if threads_in_warp == 32:
        from ... import _group_reduce as _group_reduce_frontend
        from ..._thread_group import this_warp

        return _group_reduce_frontend._reduce(
            this_warp(),
            value,
            broadcast=valid_items is None,
            valid_items=valid_items,
            **kwargs,
        )

    from . import _provider

    return _provider.provider_sum(
        value=value,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
    )


def _reduce_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    binary_op: Any = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    **kwargs: Any,
) -> Any:
    if args:
        if len(args) != 1 or valid_items is not None:
            raise TypeError(
                f"{_SCOPE}.reduce currently expects one positional value and "
                "at most one valid_items"
            )
        valid_items = args[0]
        args = ()
    validate_no_extra_args(
        "reduce",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value and optional valid_items",
    )
    threads_in_warp = _resolve_threads_in_warp(_SCOPE, "reduce", threads_in_warp)
    op = _normalize_reduce_op(binary_op)

    if threads_in_warp == 32:
        from ... import _group_reduce as _group_reduce_frontend
        from ..._thread_group import this_warp

        return _group_reduce_frontend._reduce(
            this_warp(),
            value,
            binary_op=op,
            broadcast=valid_items is None,
            valid_items=valid_items,
            **kwargs,
        )

    from . import _provider

    return _provider.provider_reduce(
        value=value,
        op=op,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
    )


def _min_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    threads_in_warp: int = 32,
    valid_items: Any = None,
    **kwargs: Any,
) -> Any:
    if args:
        if len(args) != 1 or valid_items is not None:
            raise TypeError(
                f"{_SCOPE}.min currently expects one positional value and "
                "at most one valid_items"
            )
        valid_items = args[0]
        args = ()
    validate_no_extra_args(
        "min",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value and optional valid_items",
    )
    threads_in_warp = _resolve_threads_in_warp(_SCOPE, "min", threads_in_warp)

    if threads_in_warp == 32:
        from ... import _group_reduce as _group_reduce_frontend
        from ..._thread_group import this_warp

        return _group_reduce_frontend._reduce(
            this_warp(),
            value,
            binary_op="min",
            broadcast=valid_items is None,
            valid_items=valid_items,
            **kwargs,
        )

    from . import _provider

    return _provider.provider_reduce(
        value=value,
        op="min",
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
    )


def _max_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    threads_in_warp: int = 32,
    valid_items: Any = None,
    **kwargs: Any,
) -> Any:
    if args:
        if len(args) != 1 or valid_items is not None:
            raise TypeError(
                f"{_SCOPE}.max currently expects one positional value and "
                "at most one valid_items"
            )
        valid_items = args[0]
        args = ()
    validate_no_extra_args(
        "max",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value and optional valid_items",
    )
    threads_in_warp = _resolve_threads_in_warp(_SCOPE, "max", threads_in_warp)

    if threads_in_warp == 32:
        from ... import _group_reduce as _group_reduce_frontend
        from ..._thread_group import this_warp

        return _group_reduce_frontend._reduce(
            this_warp(),
            value,
            binary_op="max",
            broadcast=valid_items is None,
            valid_items=valid_items,
            **kwargs,
        )

    from . import _provider

    return _provider.provider_reduce(
        value=value,
        op="max",
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
    )


def sum(
    value: Any,
    /,
    *args: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    **kwargs: Any,
) -> Any:
    """Return a warp-wide sum through the shared group Reduce plan.

    A physical 32-lane warp uses broadcasted CUDAX for a full reduction and
    direct CUB for a partial count. The full result is defined on every lane;
    the partial result is defined only on lane zero. Smaller logical warps keep
    the advanced scoped CUB path, whose result is defined only on logical lane
    zero. ``threads_in_warp`` must be a power of two in ``[1, 32]``.
    """
    structural_payload = {
        "value": value,
        "args": args,
        "threads_in_warp": threads_in_warp,
    }
    if valid_items is not None:
        structural_payload["valid_items"] = valid_items
    payload = merge_payload(
        "sum",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("sum", kwargs=payload)


def reduce(
    value: Any,
    /,
    *args: Any,
    binary_op: Any = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    **kwargs: Any,
) -> Any:
    """Return a warp reduction through the shared physical-warp group plan.

    ``binary_op`` accepts the same built-in aliases as
    :func:`cuda.coop.cutlass._block.reduce`: sum, multiplies, min, max,
    bit_and, bit_or, and bit_xor. Full physical-warp results are broadcast;
    partial and logical-subwarp results are defined only on lane zero.
    Arbitrary Python callables are not lowered by the CuTe warp provider yet.
    """
    structural_payload = {
        "value": value,
        "args": args,
        "binary_op": binary_op,
        "threads_in_warp": threads_in_warp,
    }
    if valid_items is not None:
        structural_payload["valid_items"] = valid_items
    payload = merge_payload(
        "reduce",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("reduce", kwargs=payload)


def min(
    value: Any,
    /,
    *args: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    **kwargs: Any,
) -> Any:
    """Return a warp-wide minimum.

    A full physical warp uses broadcasted CUDAX and defines the result on every
    lane. Partial physical-warp and logical-subwarp results use CUB and are
    defined only on lane zero of the participating warp.
    """
    structural_payload = {
        "value": value,
        "args": args,
        "threads_in_warp": threads_in_warp,
    }
    if valid_items is not None:
        structural_payload["valid_items"] = valid_items
    payload = merge_payload(
        "min",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("min", kwargs=payload)


def max(
    value: Any,
    /,
    *args: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    **kwargs: Any,
) -> Any:
    """Return a warp-wide maximum.

    A full physical warp uses broadcasted CUDAX and defines the result on every
    lane. Partial physical-warp and logical-subwarp results use CUB and are
    defined only on lane zero of the participating warp.
    """
    structural_payload = {
        "value": value,
        "args": args,
        "threads_in_warp": threads_in_warp,
    }
    if valid_items is not None:
        structural_payload["valid_items"] = valid_items
    payload = merge_payload(
        "max",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("max", kwargs=payload)


_max_provider._supports_native_thread_data = True
_min_provider._supports_native_thread_data = True
_reduce_provider._supports_native_thread_data = True
_sum_provider._supports_native_thread_data = True
_max_provider._preserves_launch_metadata = True
_min_provider._preserves_launch_metadata = True
_reduce_provider._preserves_launch_metadata = True
_sum_provider._preserves_launch_metadata = True

register_primitive_impl("max", impl=_max_provider)
register_primitive_impl("min", impl=_min_provider)
register_primitive_impl("reduce", impl=_reduce_provider)
register_primitive_impl("sum", impl=_sum_provider)
