# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar types and typed provider rendering contracts for CUTLASS."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from cutlass.base_dsl.typing import (
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
)

from cuda.coop._core.api._dispatch import _common_root_operation_name
from cuda.coop._core.dtype_policy import validate_common_numeric_dtype_name

from .._thread_data import ThreadData

ROOT_SCOPE = "cuda.coop.cutlass"


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
    Int8: TypeSpec("signed char", "i8", 8, "0"),
    Uint8: TypeSpec("unsigned char", "u8", 8, "0u"),
    Int16: TypeSpec("short", "i16", 16, "0"),
    Uint16: TypeSpec("unsigned short", "u16", 16, "0u"),
    Int32: TypeSpec("int", "i32", 32, "0"),
    Uint32: TypeSpec("unsigned int", "u32", 32, "0u"),
    Int64: TypeSpec("long long", "i64", 64, "0ll"),
    Uint64: TypeSpec("unsigned long long", "u64", 64, "0ull"),
    Float32: TypeSpec("float", "f32", 32, "0.0f"),
    Float64: TypeSpec("double", "f64", 64, "0.0"),
}
ALL_PROVIDER_TYPES = frozenset(TYPE_SPECS)
INTEGER_VALUE_TYPES = frozenset(
    value_type for value_type in TYPE_SPECS if value_type not in {Float32, Float64}
)
ORDINARY_PROVIDER_TYPES = {
    int: Int32,
    float: Float32,
    np.int8: Int8,
    np.uint8: Uint8,
    np.int16: Int16,
    np.uint16: Uint16,
    np.int32: Int32,
    np.uint32: Uint32,
    np.int64: Int64,
    np.uint64: Uint64,
    np.float32: Float32,
    np.float64: Float64,
}
PROVIDER_TYPE_NAMES = {
    value_type: value_type.__name__.lower() for value_type in TYPE_SPECS
}
_INTEGER_TYPE_TOKENS = frozenset(
    TYPE_SPECS[value_type].token for value_type in INTEGER_VALUE_TYPES
)
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
        if token in _FLOAT_TYPE_TOKENS:
            numpy_type = np.float32 if token == "f32" else np.float64
            limit = float(np.finfo(numpy_type).max)
            if not -limit <= value <= limit:
                raise ValueError(
                    f"{scope}.{name}={value} is not representable in {value_type.__name__}"
                )
            return value_type(value) if convert else value
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


def as_int32(value: Any) -> Any:
    if isinstance(value, Int32):
        return value
    return Int32(value)


def as_valid_items_arg(value: Any, *, scope: str) -> Any:
    if value is None:
        return Int32(-1)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{scope} valid_items must have an integer dtype")
    if isinstance(value, Integral):
        if not 0 <= int(value) <= (1 << 31) - 1:
            raise ValueError(f"{scope} valid_items must be between 0 and 2147483647")
        return Int32(int(value))
    if isinstance(value, Int32):
        return value
    value_type = canonical_dsl_type(value)
    if value_type not in INTEGER_VALUE_TYPES:
        raise TypeError(f"{scope} valid_items must have an integer dtype")
    if value_type in {Int8, Uint8, Int16, Uint16}:
        return Int32(value)

    from cutlass._mlir.dialects import arith

    # Preserve the range check in the source width. The provider rejects -1;
    # otherwise a wide count could wrap into an apparently valid Int32 count.
    in_range = value <= value_type((1 << 31) - 1)
    if value_type is Int64:
        in_range = in_range & (value >= value_type(0))
    return Int32(
        arith.select(
            in_range.ir_value(),
            Int32(value).ir_value(),
            Int32(-1).ir_value(),
        )
    )


def type_size_bytes(value_type: type) -> int:
    return max(1, (TYPE_SPECS[value_type].width_bits + 7) // 8)


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
    if value_type not in INTEGER_VALUE_TYPES:
        return False
    if getattr(item, "signed", None) is not None:
        return False
    mlir_type = getattr(item, "type", None)
    if mlir_type is None:
        return False
    return str(mlir_type) == f"i{TYPE_SPECS[value_type].width_bits}"


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


def canonical_dsl_type(
    value: Any,
    *,
    scope: str = ROOT_SCOPE,
    root_scope: str = ROOT_SCOPE,
) -> type:
    if isinstance(value, type) and value in TYPE_SPECS:
        return value

    if isinstance(value, np.dtype):
        ordinary_type = ORDINARY_PROVIDER_TYPES.get(value.type)
        if ordinary_type is not None:
            return ordinary_type

    if isinstance(value, type):
        ordinary_type = ORDINARY_PROVIDER_TYPES.get(value)
        if ordinary_type is not None:
            return ordinary_type

    value_type = type(value)
    ordinary_type = ORDINARY_PROVIDER_TYPES.get(value_type)
    if ordinary_type is not None:
        return ordinary_type
    if value_type in TYPE_SPECS:
        return value_type

    dsl_dtype = getattr(value, "dtype", None)
    if isinstance(dsl_dtype, type) and dsl_dtype in TYPE_SPECS:
        return dsl_dtype

    mlir_type = getattr(value, "type", None)
    if mlir_type is None:
        return value_type

    ty_str = str(mlir_type)
    signed = getattr(value, "signed", None)
    is_float = bool(getattr(value, "is_float", False))

    if is_float:
        if ty_str == "f32":
            return Float32
        if ty_str == "f64":
            return Float64
        return value_type

    if ty_str in {"i8", "i16", "i32", "i64"} and signed is None:
        raise TypeError(
            f"{scope} provider cannot infer integer signedness; "
            f"pass a {root_scope}.ThreadData value, a CUDA DSL typing class, "
            "or a value with signed=True/False"
        )
    integer_types = {
        "i8": (Int8, Uint8),
        "i16": (Int16, Uint16),
        "i32": (Int32, Uint32),
        "i64": (Int64, Uint64),
    }
    if ty_str in integer_types and isinstance(signed, bool):
        return integer_types[ty_str][0 if signed else 1]
    return value_type


def _validate_common_root_numeric_dtype(
    value: Any,
    *,
    operation: str | None = None,
) -> type:
    """Validate one CUTLASS value only while it crosses the common root."""

    value_type = canonical_dsl_type(value)
    if operation is None:
        operation = _common_root_operation_name()
    if operation is None:
        return value_type
    dtype_name = PROVIDER_TYPE_NAMES.get(value_type, "unsupported")
    validate_common_numeric_dtype_name(dtype_name, operation=operation)
    return value_type


def resolve_provider_type(
    value: Any,
    *,
    allowed: frozenset[type],
    feature: str,
    root_scope: str,
    namespace: str,
    canonical_type: Callable[[Any], type],
) -> type:
    value_type = canonical_type(value)
    operation = _common_root_operation_name()
    if operation is not None:
        dtype_name = PROVIDER_TYPE_NAMES.get(value_type, "unsupported")
        validate_common_numeric_dtype_name(dtype_name, operation=operation)
    if value_type not in TYPE_SPECS or value_type not in allowed:
        raise NotImplementedError(
            f"{root_scope}.{namespace} provider {feature} currently supports "
            f"{supported_names(allowed)} only"
        )
    return value_type


def make_provider_type_resolver(
    *,
    scope: str,
    root_scope: str,
    namespace: str,
) -> Callable[..., type]:
    def resolve_type(
        value: Any,
        *,
        allowed: frozenset[type],
        feature: str,
    ) -> type:
        return resolve_provider_type(
            value,
            allowed=allowed,
            feature=feature,
            root_scope=root_scope,
            namespace=namespace,
            canonical_type=lambda value: canonical_dsl_type(
                value,
                scope=scope,
                root_scope=root_scope,
            ),
        )

    return resolve_type
