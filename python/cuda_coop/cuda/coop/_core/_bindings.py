# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral static and runtime scalar argument bindings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._types import INT32, ArgumentKind, CxxFunction, RuntimeValue, Value


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
        return CxxFunction(str(omitted_value), INT32, name=name)
    if option.kind is BindingKind.RUNTIME:
        return Value(INT32, name=name)
    return CxxFunction(str(int(option.value)), INT32, name=name)


__all__ = ["ArgumentBinding", "BindingKind", "binding", "i32_parameter"]
