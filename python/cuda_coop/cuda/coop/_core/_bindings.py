# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile-time and runtime argument binding descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class BindingKind(str, Enum):
    """How a cooperative primitive argument is supplied."""

    OMITTED = "omitted"
    STATIC = "static"
    RUNTIME = "runtime"


@dataclass(frozen=True)
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


__all__ = ["ArgumentBinding", "BindingKind"]
