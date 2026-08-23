# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scope-aware helpers for CuTe cooperative primitive modules."""

from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any

from .._payload import reject_payload_selector_keyword
from ._launch import (
    LAUNCH_METADATA_KEYS,
    normalize_block_launch_kwargs,
    resolve_threads_in_warp,
)

ROOT_SCOPE = __name__.split("._dsl.", 1)[0]
BLOCK_SCOPE = f"{ROOT_SCOPE}._block"
WARP_SCOPE = f"{ROOT_SCOPE}._warp"
BLOCK_LAUNCH_KWARGS = (*LAUNCH_METADATA_KEYS, "threads_per_block", "dim")
_SUPPORTED_OPS_TEXT = "sum, multiplies, min, max, bit_and, bit_or, and bit_xor"
_REDUCE_OP_ALIASES = {
    None: "sum",
    "+": "sum",
    "sum": "sum",
    "add": "sum",
    "plus": "sum",
    "*": "multiplies",
    "mul": "multiplies",
    "multiply": "multiplies",
    "multiplies": "multiplies",
    "min": "min",
    "minimum": "min",
    "max": "max",
    "maximum": "max",
    "&": "bit_and",
    "bit_and": "bit_and",
    "|": "bit_or",
    "bit_or": "bit_or",
    "^": "bit_xor",
    "bit_xor": "bit_xor",
}
_CALLABLE_REDUCE_OP_ALIASES = {
    operator.add: "sum",
    operator.mul: "multiplies",
    operator.and_: "bit_and",
    operator.or_: "bit_or",
    operator.xor: "bit_xor",
}
_CALLABLE_REDUCE_OP_NAME_ALIASES = {
    ("_operator", "add"): "sum",
    ("_operator", "mul"): "multiplies",
    ("_operator", "and_"): "bit_and",
    ("_operator", "or_"): "bit_or",
    ("_operator", "xor"): "bit_xor",
    ("operator", "add"): "sum",
    ("operator", "mul"): "multiplies",
    ("operator", "and_"): "bit_and",
    ("operator", "or_"): "bit_or",
    ("operator", "xor"): "bit_xor",
    ("numpy", "add"): "sum",
    ("numpy", "multiply"): "multiplies",
    ("numpy", "minimum"): "min",
    ("numpy", "maximum"): "max",
    ("numpy", "bitwise_and"): "bit_and",
    ("numpy", "bitwise_or"): "bit_or",
    ("numpy", "bitwise_xor"): "bit_xor",
}


def scope_name(scope: str) -> str:
    return f"{ROOT_SCOPE}.{scope}"


def merge_payload(
    scope: str,
    primitive_name: str,
    structural_payload: dict[str, Any],
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    reject_payload_selector_keyword(
        extra_kwargs,
        scope=scope,
        primitive_name=primitive_name,
    )
    reserved = structural_payload.keys() & extra_kwargs.keys()
    if reserved:
        reserved_names = ", ".join(sorted(reserved))
        raise TypeError(
            f"{scope}.{primitive_name} got reserved keyword argument(s): "
            f"{reserved_names}"
        )

    payload = dict(structural_payload)
    payload.update(extra_kwargs)
    return payload


def merge_block_payload(
    primitive_name: str,
    structural_payload: dict[str, Any],
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return merge_payload(
        BLOCK_SCOPE,
        primitive_name,
        structural_payload,
        normalize_block_launch_kwargs(BLOCK_SCOPE, primitive_name, extra_kwargs),
    )


def merge_warp_payload(
    primitive_name: str,
    structural_payload: dict[str, Any],
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    payload = merge_payload(
        WARP_SCOPE, primitive_name, structural_payload, extra_kwargs
    )
    if "threads_in_warp" in payload:
        payload["threads_in_warp"] = resolve_threads_in_warp(
            WARP_SCOPE,
            primitive_name,
            payload["threads_in_warp"],
        )
    return payload


def normalize_reduce_op(
    binary_op: Any,
    *,
    scope: str,
    primitive_name: str = "reduce",
) -> str:
    try:
        return _REDUCE_OP_ALIASES[binary_op]
    except (KeyError, TypeError):
        pass

    try:
        return _CALLABLE_REDUCE_OP_ALIASES[binary_op]
    except (KeyError, TypeError):
        pass

    if callable(binary_op):
        module = getattr(binary_op, "__module__", "")
        name = getattr(binary_op, "__name__", "")
        try:
            return _CALLABLE_REDUCE_OP_NAME_ALIASES[(module, name)]
        except KeyError:
            pass

    raise NotImplementedError(
        f"{scope}.{primitive_name} currently supports {_SUPPORTED_OPS_TEXT} reductions"
    )


def normalize_scan_op(
    scan_op: Any,
    *,
    scope: str,
    primitive_name: str = "scan",
) -> str:
    try:
        return normalize_reduce_op(
            scan_op,
            scope=scope,
            primitive_name=primitive_name,
        )
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"{scope}.{primitive_name} currently supports {_SUPPORTED_OPS_TEXT} scans"
        ) from exc


def validate_default_reduce_op(
    binary_op: Any,
    *,
    scope: str,
    primitive_name: str = "reduce",
) -> None:
    if (
        normalize_reduce_op(
            binary_op,
            scope=scope,
            primitive_name=primitive_name,
        )
        == "sum"
    ):
        return
    raise NotImplementedError(
        f"{scope}.{primitive_name} currently supports only the default sum reduction"
    )


def validate_no_extra_args(
    scope: str,
    primitive_name: str,
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    expected: str,
    allowed_kwargs: Iterable[str] = (),
) -> None:
    if args:
        raise TypeError(f"{scope}.{primitive_name} {expected}")

    unexpected_kwargs = kwargs.keys() - set(allowed_kwargs)
    if unexpected_kwargs:
        unexpected = ", ".join(sorted(unexpected_kwargs))
        raise TypeError(
            f"{scope}.{primitive_name} {expected}; got unexpected keyword "
            f"argument(s): {unexpected}"
        )


def validate_no_extra_block_args(
    primitive_name: str,
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    expected: str,
) -> None:
    validate_no_extra_args(
        BLOCK_SCOPE,
        primitive_name,
        args=args,
        kwargs=kwargs,
        expected=expected,
        allowed_kwargs=BLOCK_LAUNCH_KWARGS,
    )


def validate_no_extra_warp_args(
    primitive_name: str,
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    expected: str,
) -> None:
    validate_no_extra_args(
        WARP_SCOPE,
        primitive_name,
        args=args,
        kwargs=kwargs,
        expected=expected,
        allowed_kwargs=LAUNCH_METADATA_KEYS,
    )
