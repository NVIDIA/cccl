# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""Compiler value models, dtype policy, and operation normalization."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from cutlass.base_dsl.typing import (
    Float32,
    Float64,
    Int32,
    Int64,
    Uint8,
    Uint32,
    Uint64,
)

from .._thread_data import ThreadData

ROOT_SCOPE = "cuda.coop.cutlass"

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


def merge_payload(
    scope: str,
    primitive_name: str,
    structural_payload: dict[str, Any],
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    if "payload" in extra_kwargs:
        raise TypeError(f"{scope}.{primitive_name} does not accept a payload selector")
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


@dataclass(frozen=True)
class TypeSpec:
    cpp_type: str
    token: str
    width_bits: int
    zero_literal: str


@dataclass(frozen=True)
class BundleRenderer:
    include_lines: tuple[str, ...]
    cccl_headers: tuple[tuple[str, str], ...]
    render: Callable[[Any], list[str]]
    scratch_layout_probe: Callable[[Any], ScratchLayoutProbe | None] | None = None


@dataclass(frozen=True)
class ScratchLayout:
    """Exact C++ temporary-storage layout for one specialization."""

    size_in_bytes: int
    alignment: int


@dataclass(frozen=True)
class ScratchLayoutProbe:
    """C++ constant expressions for one exact scratch layout."""

    requirement_key: Hashable
    size_expression: str
    alignment_expression: str


@dataclass(frozen=True)
class DeferredTempStorageEvent:
    """One traced cooperative call whose scratch operands need finalization."""

    kernel_op: Any
    kernel_name: str
    temp_storage: Any
    primitive_name: str
    requirement_key: Hashable
    sharing: str
    auto_sync: bool
    capacity_size_in_bytes: int | None
    capacity_alignment: int | None
    smem_addr_placeholder: Any
    size_placeholder: Any
    location: str


@dataclass(frozen=True)
class DeferredTempStorageBinding:
    """Resolved per-call scratch slice within a deferred storage plan."""

    event: DeferredTempStorageEvent
    byte_offset_in_bytes: int
    size_in_bytes: int
    alignment: int


@dataclass(frozen=True)
class DeferredTempStoragePlan:
    """One kernel-local allocation for one TempStorage identity."""

    kernel_op: Any
    kernel_name: str
    temp_storage: Any
    size_in_bytes: int
    alignment: int
    bindings: tuple[DeferredTempStorageBinding, ...]


TYPE_SPECS: dict[type, TypeSpec] = {
    Uint8: TypeSpec(
        cpp_type="unsigned char",
        token="u8",
        width_bits=8,
        zero_literal="0u",
    ),
    Int32: TypeSpec(
        cpp_type="int",
        token="i32",
        width_bits=32,
        zero_literal="0",
    ),
    Uint32: TypeSpec(
        cpp_type="unsigned int",
        token="u32",
        width_bits=32,
        zero_literal="0u",
    ),
    Int64: TypeSpec(
        cpp_type="long long",
        token="i64",
        width_bits=64,
        zero_literal="0ll",
    ),
    Uint64: TypeSpec(
        cpp_type="unsigned long long",
        token="u64",
        width_bits=64,
        zero_literal="0ull",
    ),
    Float32: TypeSpec(
        cpp_type="float",
        token="f32",
        width_bits=32,
        zero_literal="0.0f",
    ),
    Float64: TypeSpec(
        cpp_type="double",
        token="f64",
        width_bits=64,
        zero_literal="0.0",
    ),
}

SCAN_REDUCE_TYPES = frozenset(TYPE_SPECS.keys())

RADIX_KEY_TYPES = frozenset({Int32, Uint32, Int64, Uint64})

ALL_PROVIDER_TYPES = frozenset(TYPE_SPECS.keys())

ORDINARY_PROVIDER_TYPES = {
    int: Int32,
    float: Float32,
    np.uint8: Uint8,
    np.int32: Int32,
    np.uint32: Uint32,
    np.int64: Int64,
    np.uint64: Uint64,
    np.float32: Float32,
    np.float64: Float64,
}

PROVIDER_TYPE_NAMES = {
    Uint8: "uint8",
    Int32: "int32",
    Uint32: "uint32",
    Int64: "int64",
    Uint64: "uint64",
    Float32: "float32",
    Float64: "float64",
}

_INTEGER_TYPE_TOKENS = frozenset({"u8", "i32", "u32", "i64", "u64"})

_FLOAT_TYPE_TOKENS = frozenset({"f32", "f64"})

_NOT_PLAIN_SCALAR = object()


def supported_names(types: frozenset[type]) -> str:
    return "/".join(sorted(t.__name__ for t in types))


def coerce_plain_scalar(
    value: Any,
    value_type: type,
    *,
    name: str,
    scope: str,
    allow_nonfinite: bool,
    convert: bool = True,
) -> Any:
    """Validate and optionally convert an exact Python numeric literal."""

    token = TYPE_SPECS[value_type].token
    if type(value) is int:
        if token not in _INTEGER_TYPE_TOKENS:
            raise TypeError(
                f"{scope}.{name} dtype does not match {value_type.__name__}"
            )
        bits = int(token.lstrip("iu"))
        lower = 0 if token.startswith("u") else -(1 << (bits - 1))
        upper = (1 << bits) - 1 if token.startswith("u") else (1 << (bits - 1)) - 1
        if not lower <= value <= upper:
            raise ValueError(
                f"{scope}.{name}={value} is not representable in {value_type.__name__}"
            )
        return value_type(value) if convert else value
    if type(value) is float:
        if token not in _FLOAT_TYPE_TOKENS:
            raise TypeError(
                f"{scope}.{name} dtype does not match {value_type.__name__}"
            )
        if not allow_nonfinite and not math.isfinite(value):
            raise ValueError(f"{scope}.{name} must be finite")
        numpy_type = np.float32 if token == "f32" else np.float64
        limit = float(np.finfo(numpy_type).max)
        if math.isfinite(value) and abs(value) > limit:
            raise ValueError(
                f"{scope}.{name}={value} is not representable in {value_type.__name__}"
            )
        return value_type(value) if convert else value
    return _NOT_PLAIN_SCALAR


def validate_scan_reduce_op_for_type(
    op: str,
    value_type: type,
    *,
    root_scope: str,
    feature: str,
    namespace: str = "block",
) -> None:
    if op.startswith("bit_") and value_type not in RADIX_KEY_TYPES:
        operation = "reductions" if feature == "reduce" else "operations"
        raise TypeError(
            f"{root_scope}.{namespace}.{feature} bitwise {operation} require "
            "an integral type"
        )


def reduce_op_expr(op: str, lhs: str, rhs: str) -> str:
    if op == "sum":
        return f"{lhs} + {rhs}"
    if op == "multiplies":
        return f"{lhs} * {rhs}"
    if op == "min":
        return f"(({rhs}) < ({lhs}) ? ({rhs}) : ({lhs}))"
    if op == "max":
        return f"(({rhs}) > ({lhs}) ? ({rhs}) : ({lhs}))"
    if op == "bit_and":
        return f"{lhs} & {rhs}"
    if op == "bit_or":
        return f"{lhs} | {rhs}"
    if op == "bit_xor":
        return f"{lhs} ^ {rhs}"
    raise NotImplementedError(f"Unsupported reduce op: {op}")


def cub_op_expr(op: str) -> str:
    if op == "sum":
        return "::cuda::std::plus<>{}"
    if op == "multiplies":
        return "::cuda::std::multiplies<>{}"
    if op == "min":
        return "::cuda::minimum<>{}"
    if op == "max":
        return "::cuda::maximum<>{}"
    if op == "bit_and":
        return "::cuda::std::bit_and<>{}"
    if op == "bit_or":
        return "::cuda::std::bit_or<>{}"
    if op == "bit_xor":
        return "::cuda::std::bit_xor<>{}"
    raise NotImplementedError(f"Unsupported CUB op: {op}")


def as_int32(value: Any) -> Any:
    if isinstance(value, Int32):
        return value
    return Int32(value)


def as_valid_items_arg(value: Any, *, scope: str) -> Any:
    if value is None:
        return Int32(-1)
    if isinstance(value, Int32):
        return value
    try:
        return Int32(value)
    except Exception as exc:
        raise TypeError(f"{scope} valid_items must be convertible to Int32") from exc


def _static_int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        raise TypeError("radix bit bounds must be int-like scalars")
    if isinstance(value, Integral):
        return int(value)
    return None


def validate_radix_bit_range(
    begin_bit: Any,
    end_bit: Any | None,
    key_type: type,
) -> Any:
    width_bits = TYPE_SPECS[key_type].width_bits
    resolved_end_bit = width_bits if end_bit is None else end_bit

    static_begin_bit = _static_int_value(begin_bit)
    static_end_bit = _static_int_value(resolved_end_bit)

    if static_begin_bit is not None and static_begin_bit < 0:
        raise ValueError("begin_bit must be non-negative")
    if static_begin_bit is not None and static_begin_bit >= width_bits:
        raise ValueError(f"begin_bit must be < {width_bits}")
    if (
        static_begin_bit is not None
        and static_end_bit is not None
        and static_end_bit <= static_begin_bit
    ):
        raise ValueError("end_bit must be greater than begin_bit")
    if static_end_bit is not None and static_end_bit > width_bits:
        raise ValueError(f"end_bit must be <= {width_bits}")
    return resolved_end_bit


def type_size_bytes(value_type: type) -> int:
    return max(1, (TYPE_SPECS[value_type].width_bits + 7) // 8)


def coerce_scan_initial_value(
    *,
    initial_value: Any,
    value_type: type,
    root_scope: str,
    feature: str,
    namespace: str,
) -> Any:
    if isinstance(initial_value, value_type):
        return initial_value
    try:
        return value_type(initial_value)
    except Exception as exc:
        raise TypeError(
            f"{root_scope}.{namespace}.{feature} initial_value cannot be converted to "
            f"{value_type.__name__}"
        ) from exc


def resolve_thread_data_value_type(
    value: ThreadData,
    *,
    allowed: frozenset[type],
    feature: str,
    scope: str,
    resolve_type: Callable[..., type],
    supported_types: frozenset[type] = ALL_PROVIDER_TYPES,
) -> tuple[type, tuple[Any, ...]]:
    values = value.values(feature)
    if value.dtype is not None:
        value_type = resolve_type(value.dtype, allowed=allowed, feature=feature)
        converted: list[Any] = []
        for idx, item in enumerate(values):
            plain_item = coerce_plain_scalar(
                item,
                value_type,
                name=f"{feature} ThreadData item {idx}",
                scope=scope,
                allow_nonfinite=True,
            )
            if plain_item is not _NOT_PLAIN_SCALAR:
                converted.append(plain_item)
                continue
            try:
                item_type = resolve_type(
                    item,
                    allowed=supported_types,
                    feature=feature,
                )
            except TypeError as exc:
                if _signless_integer_item_matches_dtype(item, value_type):
                    converted.append(item)
                    continue
                raise TypeError(
                    f"{scope}.{feature} ThreadData item {idx} type "
                    "cannot be reconciled with declared dtype"
                ) from exc
            except NotImplementedError as exc:
                raise TypeError(
                    f"{scope}.{feature} ThreadData item {idx} type "
                    "cannot be reconciled with declared dtype"
                ) from exc
            if item_type is not value_type:
                raise TypeError(
                    f"{scope}.{feature} ThreadData dtype does not match "
                    "initialized item types"
                )
            converted.append(item)
        return value_type, tuple(converted)

    value_type = resolve_type(values[0], allowed=allowed, feature=feature)
    for item in values[1:]:
        item_type = resolve_type(item, allowed=allowed, feature=feature)
        if item_type is not value_type:
            raise TypeError(
                f"{scope}.{feature} ThreadData requires homogeneous item types"
            )
    return value_type, values


def _signless_integer_item_matches_dtype(item: Any, value_type: type) -> bool:
    if value_type not in {Int32, Uint32, Int64, Uint64}:
        return False
    if getattr(item, "signed", None) is not None:
        return False
    mlir_type = getattr(item, "type", None)
    if mlir_type is None:
        return False
    return str(mlir_type) == f"i{TYPE_SPECS[value_type].width_bits}"


def resolve_thread_data_pair_types(
    *,
    key: Any,
    value: Any,
    allowed_key_types: frozenset[type],
    allowed_value_types: frozenset[type],
    feature: str,
    scope: str,
    resolve_type: Callable[..., type],
) -> tuple[type, tuple[Any, ...], ThreadData, type, tuple[Any, ...], ThreadData]:
    if isinstance(key, ThreadData) or isinstance(value, ThreadData):
        if not isinstance(key, ThreadData) or not isinstance(value, ThreadData):
            raise TypeError(
                f"{scope}.{feature} requires both key and value to be "
                "ThreadData when one argument uses ThreadData"
            )
        if key.items_per_thread != value.items_per_thread:
            raise ValueError(
                f"{scope}.{feature} requires matching "
                "ThreadData.items_per_thread for key and value"
            )

    if not isinstance(key, ThreadData) or not isinstance(value, ThreadData):
        raise TypeError(
            f"{scope}.{feature} internal ThreadData path requires "
            "ThreadData key/value inputs"
        )

    key_type, key_values = resolve_thread_data_value_type(
        key,
        allowed=allowed_key_types,
        feature=feature,
        scope=scope,
        resolve_type=resolve_type,
    )
    value_type, value_values = resolve_thread_data_value_type(
        value,
        allowed=allowed_value_types,
        feature=feature,
        scope=scope,
        resolve_type=resolve_type,
    )
    return key_type, key_values, key, value_type, value_values, value


def validate_thread_data_output(
    *,
    output: Any,
    expected_items_per_thread: int,
    resolved_dtype: type,
    scope: str,
    primitive_name: str,
    output_name: str,
    resolve_type: Callable[..., type],
    assigned_dtype: Any | None = None,
    type_label: str = "ThreadData",
    item_count_message: str | None = None,
) -> ThreadData | None:
    if output is None:
        return None
    if not isinstance(output, ThreadData):
        raise TypeError(f"{scope}.{primitive_name} {output_name} must be {type_label}")
    if output.items_per_thread != expected_items_per_thread:
        if item_count_message is None:
            item_count_message = (
                f"{scope}.{primitive_name} {output_name} must have "
                f"items_per_thread={expected_items_per_thread}"
            )
        raise ValueError(item_count_message)
    if output.dtype is not None:
        resolve_type(
            output.dtype,
            allowed=frozenset({resolved_dtype}),
            feature=primitive_name,
        )
    else:
        output.dtype = resolved_dtype if assigned_dtype is None else assigned_dtype
    return output


def thread_data_output_dtype(value: ThreadData, value_type: type) -> Any:
    return value.dtype if value.dtype is not None else value_type
