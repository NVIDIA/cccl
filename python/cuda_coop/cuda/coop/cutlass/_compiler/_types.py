# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reconcile payload values, common API dtypes, and the provider's C++ ABI.

Ordinary Python and NumPy types map to CUTLASS scalar types. A bare signless
integer IR value needs dtype metadata to recover signedness. A common-root
call is a ``cuda.coop.*`` call that the common API delegated to this backend.
Such calls must also pass the common API's dtype checks.
"""

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
class TypeSpecification:
    """Describe a scalar type for the C++ wrapper and its symbol."""

    cpp_type: str
    token: str
    width_bits: int
    zero_literal: str


@dataclass(frozen=True)
class BundleRenderer:
    """Associate a provider kind with its source and scratch requirements.

    The renderer supplies includes, header lookup paths, and C++ wrapper text.
    Its optional ``scratch_layout_probe`` callback returns a layout query
    when a request needs storage, or ``None`` for a storage-free request.
    """

    include_lines: tuple[str, ...]
    cccl_headers: tuple[tuple[str, str], ...]
    render: Callable[[Any], list[str]]
    scratch_layout_probe: Callable[[Any], ScratchLayoutProbe | None] | None = (
        None
    )


@dataclass(frozen=True)
class ScratchLayout:
    """Record a compiled C++ scratch size and alignment, both in bytes.

    The probe decoder validates these values before allocation planning uses
    them. This record does not allocate storage or select a sharing policy.
    """

    size_in_bytes: int
    alignment: int


@dataclass(frozen=True)
class ScratchLayoutProbe:
    """Describe a C++ layout query and the calls that will consume it.

    ``requirement_key`` joins traced uses to the result. ``size_expression``
    and ``alignment_expression`` are C++ constant expressions evaluated in the
    provider translation unit. Several keys may refer to the same expressions;
    the compiler only needs to evaluate that expression pair once.
    """

    requirement_key: Hashable
    size_expression: str
    alignment_expression: str


@dataclass(frozen=True)
class DeferredTempStorageEvent:
    """Keep one traced call's scratch operands until finalization.

    ``kernel_op`` and ``temp_storage`` identify the kernel and descriptor that
    own the allocation. ``requirement_key`` selects its compiled C++ layout.
    Sharing, synchronization, and capacity fields capture descriptor policy
    at the call. The planner checks that later uses retain the same policy.
    A typed reservation supplies ``reservation_layout`` directly instead of
    a C++ layout probe.

    ``smem_addr_placeholder`` and ``size_placeholder`` are MLIR values to
    replace after allocation. ``kernel_name``, ``primitive_name``, and
    ``location`` identify the use in diagnostics. An event represents a traced
    call site, not each runtime execution of a loop containing that call.
    """

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
    reservation_layout: ScratchLayout | None = None


@dataclass(frozen=True)
class DeferredTempStorageBinding:
    """Assign one traced call a slice within a descriptor's allocation.

    ``byte_offset_in_bytes`` is relative to the allocation base. The byte size
    and alignment describe the slice available to ``event``. Shared bindings
    use the whole allocation; exclusive bindings use their own C++ layout.
    """

    event: DeferredTempStorageEvent
    byte_offset_in_bytes: int
    size_in_bytes: int
    alignment: int


@dataclass(frozen=True)
class DeferredTempStoragePlan:
    """Collect all scratch uses of one descriptor in one kernel.

    ``size_in_bytes`` and ``alignment`` describe the allocation to insert at
    kernel entry. ``bindings`` maps its slices back to traced call operands.
    The same descriptor used in another kernel receives a separate plan.
    """

    kernel_op: Any
    kernel_name: str
    temp_storage: Any
    size_in_bytes: int
    alignment: int
    bindings: tuple[DeferredTempStorageBinding, ...]


TYPE_SPECIFICATIONS: dict[type, TypeSpecification] = {
    Int8: TypeSpecification("signed char", "i8", 8, "0"),
    Uint8: TypeSpecification("unsigned char", "u8", 8, "0u"),
    Int16: TypeSpecification("short", "i16", 16, "0"),
    Uint16: TypeSpecification("unsigned short", "u16", 16, "0u"),
    Int32: TypeSpecification("int", "i32", 32, "0"),
    Uint32: TypeSpecification("unsigned int", "u32", 32, "0u"),
    Int64: TypeSpecification("long long", "i64", 64, "0ll"),
    Uint64: TypeSpecification("unsigned long long", "u64", 64, "0ull"),
    Float32: TypeSpecification("float", "f32", 32, "0.0f"),
    Float64: TypeSpecification("double", "f64", 64, "0.0"),
}
ALL_PROVIDER_TYPES = frozenset(TYPE_SPECIFICATIONS)
INTEGER_VALUE_TYPES = frozenset(
    value_type
    for value_type in TYPE_SPECIFICATIONS
    if value_type not in {Float32, Float64}
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
    value_type: value_type.__name__.lower()
    for value_type in TYPE_SPECIFICATIONS
}
_INTEGER_TYPE_TOKENS = frozenset(
    TYPE_SPECIFICATIONS[value_type].token for value_type in INTEGER_VALUE_TYPES
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
    """Validate and optionally cast an exact Python int or float.

    Lowerings call this before forming typed wrapper arguments. A Python
    literal has no fixed-width compiler dtype, so it must be checked against
    the destination type before conversion can discard high bits.

    Check the destination range and the requested nonfinite policy. Normal
    floating-point rounding can still occur. Return a sentinel for other
    scalar objects so their declared dtype can be checked separately.

    Parameters
    ----------
    value : object
        Candidate host literal. Only exact Python int and float enter this
        conversion path; typed compiler values use the separate dtype checks.
    value_type : type
        Supported CUTLASS scalar type required by the wrapper parameter.
    name : str
        Argument name or operation detail for a conversion diagnostic.
    scope : str
        Qualified operation prefix used in diagnostics.
    allow_nonfinite : bool
        Whether floating-point infinities and NaNs are accepted.
    convert : bool
        Whether to build the CUTLASS scalar after checking its range. False
        returns the original Python literal after the same checks.

    Returns
    -------
    object
        Converted scalar, validated original literal when convert is false,
        or ``_NOT_PLAIN_SCALAR`` when the input is not an exact int or float.
    """

    token = TYPE_SPECIFICATIONS[value_type].token
    if type(value) is int:
        if token in _FLOAT_TYPE_TOKENS:
            numpy_type = np.float32 if token == "f32" else np.float64
            limit = float(np.finfo(numpy_type).max)
            if not -limit <= value <= limit:
                raise ValueError(
                    f"{scope}.{name}={value} is not representable in "
                    f"{value_type.__name__}"
                )
            return value_type(value) if convert else value
        if token not in _INTEGER_TYPE_TOKENS:
            raise TypeError(
                f"{scope}.{name} dtype does not match {value_type.__name__}"
            )
        bits = int(token.lstrip("iu"))
        lower = 0 if token.startswith("u") else -(1 << (bits - 1))
        upper = (
            (1 << bits) - 1 if token.startswith("u") else (1 << (bits - 1)) - 1
        )
        if not lower <= value <= upper:
            raise ValueError(
                f"{scope}.{name}={value} is not representable in "
                f"{value_type.__name__}"
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
                f"{scope}.{name}={value} is not representable in "
                f"{value_type.__name__}"
            )
        return value_type(value) if convert else value
    return _NOT_PLAIN_SCALAR


def as_valid_items_arg(value: Any, *, scope: str) -> Any:
    """Convert a runtime count to the provider's signed 32-bit ABI.

    Check wide DSL values before narrowing. An out-of-range value becomes -1
    so the generated provider trap rejects it instead of accepting a wrapped
    count. Reject invalid host integers directly. An omitted count has no
    extern argument, so callers should not pass None. If None does arrive,
    return -1 so the provider trap rejects it.
    """

    if value is None:
        return Int32(-1)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{scope} valid_items must have an integer dtype")
    if isinstance(value, Integral):
        if not 0 <= int(value) <= (1 << 31) - 1:
            raise ValueError(
                f"{scope} valid_items must be between 0 and 2147483647"
            )
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


def resolve_thread_data_value_type(
    value: ThreadData,
    *,
    allowed: frozenset[type],
    feature: str,
    scope: str,
    resolve_type: Callable[..., type],
    supported_types: frozenset[type] = ALL_PROVIDER_TYPES,
) -> tuple[type, tuple[Any, ...]]:
    """Resolve initialized payload items to one supported provider dtype.

    With declared metadata, convert Python literals and require typed items
    to agree. The declared dtype can supply signedness for a raw integer IR
    value of matching width. Without metadata, infer from all items and
    require a homogeneous type.
    """

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
    """Match raw IR width when dtype metadata supplies signedness."""

    if value_type not in INTEGER_VALUE_TYPES:
        return False
    if getattr(item, "signed", None) is not None:
        return False
    mlir_type = getattr(item, "type", None)
    if mlir_type is None:
        return False
    return str(mlir_type) == f"i{TYPE_SPECIFICATIONS[value_type].width_bits}"


def thread_data_output_dtype(value: ThreadData, value_type: type) -> Any:
    return value.dtype if value.dtype is not None else value_type


def canonical_dsl_type(
    value: Any,
    *,
    scope: str = ROOT_SCOPE,
    root_scope: str = ROOT_SCOPE,
) -> type:
    """Resolve values, dtype tokens, and typed IR to a CUTLASS type.

    Raise TypeError for raw i8/i16/i32/i64 values without signedness instead
    of guessing. Return the Python type of other unsupported values so the
    caller can issue an operation-specific diagnostic.
    """

    if isinstance(value, type) and value in TYPE_SPECIFICATIONS:
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
    if value_type in TYPE_SPECIFICATIONS:
        return value_type

    dsl_dtype = getattr(value, "dtype", None)
    if isinstance(dsl_dtype, type) and dsl_dtype in TYPE_SPECIFICATIONS:
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
    """Apply common API restrictions and check the provider dtype set.

    A qualified call can use its provider contract directly. A delegated
    common call must also pass the common API's dtype check.
    """

    value_type = canonical_type(value)
    operation = _common_root_operation_name()
    if operation is not None:
        dtype_name = PROVIDER_TYPE_NAMES.get(value_type, "unsupported")
        validate_common_numeric_dtype_name(dtype_name, operation=operation)
    if value_type not in TYPE_SPECIFICATIONS or value_type not in allowed:
        raise NotImplementedError(
            f"{root_scope}.{namespace} provider {feature} supports "
            f"{supported_names(allowed)} only"
        )
    return value_type


def make_provider_type_resolver(
    *,
    scope: str,
    root_scope: str,
    namespace: str,
) -> Callable[..., type]:
    """Bind operation names for consistent type diagnostics."""

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


__all__ = [
    "ALL_PROVIDER_TYPES",
    "INTEGER_VALUE_TYPES",
    "ORDINARY_PROVIDER_TYPES",
    "PROVIDER_TYPE_NAMES",
    "ROOT_SCOPE",
    "TYPE_SPECIFICATIONS",
    "_NOT_PLAIN_SCALAR",
    "BundleRenderer",
    "DeferredTempStorageBinding",
    "DeferredTempStorageEvent",
    "DeferredTempStoragePlan",
    "ScratchLayout",
    "ScratchLayoutProbe",
    "TypeSpecification",
    "Uint8",
    "Uint32",
    "Uint64",
    "_validate_common_root_numeric_dtype",
    "as_valid_items_arg",
    "canonical_dsl_type",
    "coerce_plain_scalar",
    "make_provider_type_resolver",
    "resolve_provider_type",
    "resolve_thread_data_value_type",
    "supported_names",
    "thread_data_output_dtype",
]
