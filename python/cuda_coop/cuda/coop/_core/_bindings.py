# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile-time and runtime argument binding descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any

_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1


class BindingKind(str, Enum):
    """How a cooperative primitive argument is supplied."""

    OMITTED = "omitted"
    STATIC = "static"
    RUNTIME = "runtime"


@dataclass(frozen=True, eq=False)
class ArgumentBinding:
    """Binding category and optional compile-time value for one argument."""

    kind: BindingKind
    value: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", BindingKind(self.kind))
        if self.kind is not BindingKind.STATIC and self.value is not None:
            raise ValueError("only static bindings may carry a value")

    @classmethod
    def omitted(cls) -> ArgumentBinding:
        return cls(BindingKind.OMITTED)

    @classmethod
    def static(cls, value: Any) -> ArgumentBinding:
        return cls(BindingKind.STATIC, value)

    @classmethod
    def runtime(cls) -> ArgumentBinding:
        return cls(BindingKind.RUNTIME)

    @property
    def semantic_key(self) -> tuple[str, ...]:
        """Return a type- and representation-stable binding identity."""

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


def _normalize_integer_binding(
    binding: ArgumentBinding,
    *,
    name: str,
    minimum: int,
    maximum: int,
    bits: int,
) -> ArgumentBinding:
    if binding.kind is not BindingKind.STATIC:
        return binding
    value = binding.value
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"static {name} must be an integer")
    normalized = int(value)
    if not minimum <= normalized <= maximum:
        raise ValueError(f"static {name} must fit a signed {bits}-bit integer")
    return ArgumentBinding.static(normalized)


def _normalize_i32_binding(
    binding: ArgumentBinding,
    *,
    name: str,
) -> ArgumentBinding:
    return _normalize_integer_binding(
        binding,
        name=name,
        minimum=_I32_MIN,
        maximum=_I32_MAX,
        bits=32,
    )


def _normalize_i64_binding(
    binding: ArgumentBinding,
    *,
    name: str,
) -> ArgumentBinding:
    return _normalize_integer_binding(
        binding,
        name=name,
        minimum=_I64_MIN,
        maximum=_I64_MAX,
        bits=64,
    )


__all__ = ["ArgumentBinding", "BindingKind"]
