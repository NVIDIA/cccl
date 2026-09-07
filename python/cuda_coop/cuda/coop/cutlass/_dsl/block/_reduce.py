# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from cuda.coop._core.block import normalize_block_row_reduce_geometry

from .._scope import BLOCK_SCOPE as _SCOPE
from .._scope import merge_block_payload as merge_payload
from .._scope import normalize_reduce_op as _normalize_reduce_op_impl
from .._scope import validate_default_reduce_op as _validate_default_reduce_op_impl
from .._scope import validate_no_extra_block_args as validate_no_extra_args
from .._temp_storage import _validate_block_row_reduce_launch
from ._dispatch import dispatch_primitive, register_primitive_impl


def _normalize_reduce_op(binary_op: Any) -> str:
    return _normalize_reduce_op_impl(binary_op, scope=_SCOPE)


def _validate_default_reduce_op(binary_op: Any) -> None:
    _validate_default_reduce_op_impl(binary_op, scope=_SCOPE)


def _sum_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    num_valid: Any = None,
    valid_items: Any = None,
    algorithm: Any = None,
    **kwargs: Any,
) -> Any:
    if args:
        if len(args) != 1 or num_valid is not None or valid_items is not None:
            raise TypeError(
                f"{_SCOPE}.sum currently expects one positional value and "
                "at most one num_valid"
            )
        num_valid = args[0]
        args = ()
    if num_valid is not None and valid_items is not None:
        raise TypeError(f"{_SCOPE}.sum got both num_valid and valid_items")
    if valid_items is not None:
        num_valid = valid_items
    validate_no_extra_args(
        "sum",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value and optional num_valid",
    )

    from ... import _group_reduce as _group_reduce_frontend
    from ..._thread_group import this_block

    return _group_reduce_frontend._reduce(
        this_block(),
        value,
        broadcast=num_valid is None and algorithm is None,
        valid_items=num_valid,
        algorithm=algorithm,
        **kwargs,
    )


_sum_provider._supports_native_thread_data = True
_sum_provider._preserves_launch_metadata = True
_sum_provider._uses_planned_temp_storage = True


def _row_sum_provider(
    *,
    value: Any,
    rows_per_block: int,
    warps_per_row: int,
    args: tuple[Any, ...] = (),
    **kwargs: Any,
) -> Any:
    validate_no_extra_args(
        "row_sum",
        args=args,
        kwargs=kwargs,
        expected=(
            "currently expects one positional value plus rows_per_block and "
            "warps_per_row"
        ),
    )

    geometry = normalize_block_row_reduce_geometry(
        rows_per_block=rows_per_block,
        warps_per_row=warps_per_row,
    )
    _validate_block_row_reduce_launch(geometry, kwargs, scope=_SCOPE)

    from . import _provider

    return _provider._provider_row_sum_after_launch_validation(
        value=value,
        rows_per_block=geometry.rows_per_block,
        warps_per_row=geometry.warps_per_row,
    )


_row_sum_provider._supports_native_thread_data = True
_row_sum_provider._preserves_launch_metadata = True


def _reduce_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    binary_op: Any = None,
    num_valid: Any = None,
    valid_items: Any = None,
    algorithm: Any = None,
    **kwargs: Any,
) -> Any:
    if args:
        if len(args) != 1 or num_valid is not None or valid_items is not None:
            raise TypeError(
                f"{_SCOPE}.reduce currently expects one positional value and "
                "at most one num_valid"
            )
        num_valid = args[0]
        args = ()
    if num_valid is not None and valid_items is not None:
        raise TypeError(f"{_SCOPE}.reduce got both num_valid and valid_items")
    if valid_items is not None:
        num_valid = valid_items
    validate_no_extra_args(
        "reduce",
        args=args,
        kwargs=kwargs,
        expected="currently expects one positional value and optional num_valid",
    )
    op = _normalize_reduce_op(binary_op)

    from ... import _group_reduce as _group_reduce_frontend
    from ..._thread_group import this_block

    return _group_reduce_frontend._reduce(
        this_block(),
        value,
        binary_op=op,
        broadcast=num_valid is None and algorithm is None,
        valid_items=num_valid,
        algorithm=algorithm,
        **kwargs,
    )


_reduce_provider._supports_native_thread_data = True
_reduce_provider._preserves_launch_metadata = True
_reduce_provider._uses_planned_temp_storage = True


def sum(
    value: Any,
    /,
    *args: Any,
    algorithm: Any = None,
    **kwargs: Any,
) -> Any:
    """Return a block-wide sum through the shared group Reduce plan.

    Full scalar or ``ThreadData`` input uses broadcasted CUDAX Reduce and returns
    one scalar to every thread. ``num_valid``/``valid_items`` and explicit CUB
    ``algorithm`` selection use direct CUB and return a value defined only at
    block rank zero. Partial counts are supported only for scalar input.

    A legacy ``TempStorage`` argument is accepted for compatibility, but the
    shared plan owns the exact CUDAX/CUB scratch and does not use or charge that
    object.
    """
    structural_payload = {
        "value": value,
        "args": args,
    }
    if algorithm is not None:
        structural_payload["algorithm"] = algorithm
    payload = merge_payload(
        "sum",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("sum", kwargs=payload)


def row_sum(
    value: Any,
    /,
    *args: Any,
    rows_per_block: int,
    warps_per_row: int,
    **kwargs: Any,
) -> Any:
    """Return a CUB ``BlockRowReduce`` sum for each logical row.

    ``rows_per_block`` and ``warps_per_row`` describe how the CTA is partitioned
    into row reductions. The row aggregate is broadcast to participating lanes
    by the provider shim.
    """
    payload = merge_payload(
        "row_sum",
        {
            "value": value,
            "args": args,
            "rows_per_block": rows_per_block,
            "warps_per_row": warps_per_row,
        },
        kwargs,
    )
    return dispatch_primitive("row_sum", kwargs=payload)


def reduce(
    value: Any,
    /,
    *args: Any,
    binary_op: Any = None,
    algorithm: Any = None,
    **kwargs: Any,
) -> Any:
    """Return a block-wide reduction through the shared group Reduce plan.

    ``binary_op`` accepts sum, multiplies, min, max, bit_and, bit_or, and
    bit_xor spellings, plus known ``operator``/NumPy aliases. Full scalar or
    ``ThreadData`` input is broadcast by CUDAX. ``num_valid``/``valid_items``
    and explicit CUB ``algorithm`` selection use direct CUB and return a value
    defined only at block rank zero. Partial counts are supported only for
    scalar input. Arbitrary Python callables are not lowered yet.
    """
    structural_payload = {
        "value": value,
        "args": args,
        "binary_op": binary_op,
    }
    if algorithm is not None:
        structural_payload["algorithm"] = algorithm
    payload = merge_payload(
        "reduce",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("reduce", kwargs=payload)


register_primitive_impl("reduce", impl=_reduce_provider)
register_primitive_impl("row_sum", impl=_row_sum_provider)
register_primitive_impl("sum", impl=_sum_provider)
