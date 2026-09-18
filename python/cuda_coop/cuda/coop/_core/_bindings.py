# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral static and runtime scalar argument bindings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any

from ._types import INT32, ArgumentKind, CxxFunction, RuntimeValue, Value

_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1
_U64_MAX = (1 << 64) - 1


class BindingKind(str, Enum):
    """How a factory option reaches the generated primitive call."""

    OMITTED = "omitted"
    STATIC = "static"
    RUNTIME = "runtime"


@dataclass(frozen=True, eq=False)
class ArgumentBinding:
    """An omitted, compile-time, or runtime scalar factory argument."""

    kind: BindingKind
    value: Any = None

    def __post_init__(self) -> None:
        if self.kind is not BindingKind.STATIC and self.value is not None:
            raise ValueError("only static argument bindings may carry a value")

    @classmethod
    def omitted(cls) -> "ArgumentBinding":
        return cls(BindingKind.OMITTED)

    @classmethod
    def static(cls, value: Any) -> "ArgumentBinding":
        return cls(BindingKind.STATIC, value)

    @classmethod
    def runtime(cls) -> "ArgumentBinding":
        return cls(BindingKind.RUNTIME)

    @property
    def semantic_key(self) -> tuple[str, ...]:
        """Return a type- and representation-stable request identity."""

        if self.kind is not BindingKind.STATIC:
            return (self.kind.value,)
        value_type = type(self.value)
        return (
            self.kind.value,
            value_type.__module__,
            value_type.__qualname__,
            repr(self.value),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArgumentBinding):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def argument_kind(self) -> ArgumentKind | None:
        if self.kind is BindingKind.OMITTED:
            return None
        if self.kind is BindingKind.STATIC:
            return ArgumentKind.STATIC
        return ArgumentKind.RUNTIME


def binding(value: Any, *, omitted: Any = None) -> ArgumentBinding:
    """Classify a frontend value without retaining runtime payload data."""

    if value is omitted:
        return ArgumentBinding.omitted()
    if isinstance(value, RuntimeValue):
        return ArgumentBinding.runtime()
    return ArgumentBinding.static(value)


def i32_parameter(
    option: ArgumentBinding,
    *,
    name: str,
    omitted_value: int | None = None,
) -> Value | CxxFunction | None:
    """Materialize an i32 binding as a core runtime or constant parameter."""

    if option.kind is BindingKind.OMITTED:
        if omitted_value is None:
            return None
        value = _normalize_i32(omitted_value, name=name, source="omitted")
        return CxxFunction(str(value), INT32, name=name)
    if option.kind is BindingKind.RUNTIME:
        return Value(INT32, name=name)
    value = _normalize_i32(option.value, name=name, source="static")
    return CxxFunction(str(value), INT32, name=name)


def _normalize_i32(value: Any, *, name: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{source} {name} must be an integer")
    normalized = int(value)
    if not _I32_MIN <= normalized <= _I32_MAX:
        raise ValueError(f"{source} {name} must fit a signed 32-bit integer")
    return normalized


def _normalize_i32_binding(
    option: ArgumentBinding,
    *,
    name: str,
) -> ArgumentBinding:
    """Canonicalize the value identity of one static signed-i32 binding."""

    if option.kind is not BindingKind.STATIC:
        return option
    return ArgumentBinding.static(
        _normalize_i32(option.value, name=name, source="static")
    )


def _normalize_i64(value: Any, *, name: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{source} {name} must be an integer")
    normalized = int(value)
    if not _I64_MIN <= normalized <= _I64_MAX:
        raise ValueError(f"{source} {name} must fit a signed 64-bit integer")
    return normalized


def _normalize_i64_binding(
    option: ArgumentBinding,
    *,
    name: str,
) -> ArgumentBinding:
    """Canonicalize the value identity of one static signed-i64 binding."""

    if option.kind is not BindingKind.STATIC:
        return option
    return ArgumentBinding.static(
        _normalize_i64(option.value, name=name, source="static")
    )


def _cxx_scalar_literal(value: Any, *, name: str) -> str:
    """Render one finite scalar as a C++ source literal."""

    scalar = getattr(value, "value", value)
    if isinstance(scalar, bool):
        return "true" if scalar else "false"
    if isinstance(scalar, Integral):
        normalized = int(scalar)
        if not _I64_MIN <= normalized <= _U64_MAX:
            raise ValueError(f"static {name} must fit a 64-bit integer")
        if normalized == _I64_MIN:
            return "(-9223372036854775807LL - 1LL)"
        if normalized > _I64_MAX:
            return f"{normalized}ULL"
        return str(normalized)
    if isinstance(scalar, Real):
        normalized = float(scalar)
        if not math.isfinite(normalized):
            raise ValueError(f"static {name} must be finite")
        return repr(normalized)
    raise TypeError(f"static {name} must be a numeric scalar")


__all__ = ["ArgumentBinding", "BindingKind", "binding", "i32_parameter"]
