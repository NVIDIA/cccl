# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scope-independent scan operation semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._symbols import semantic_token
from ._types import (
    CxxFunction,
    CxxOperator,
    PythonOperator,
    Reference,
    StatefulOperator,
)


class ScanMode(str, Enum):
    EXCLUSIVE = "exclusive"
    INCLUSIVE = "inclusive"


class ScanValueKind(str, Enum):
    SCALAR = "scalar"
    ARRAY = "array"


_SCAN_OPERATORS = (CxxOperator, PythonOperator, StatefulOperator)
_INITIAL_VALUES = (CxxFunction, Reference)


@dataclass(frozen=True, eq=False)
class ScanSemantics:
    dtype: Any
    mode: ScanMode
    value_kind: ScanValueKind
    items_per_thread: int
    scan_operator: CxxOperator | PythonOperator | StatefulOperator | None = None
    initial_value: CxxFunction | Reference | None = None
    aggregate: bool = False
    prefix_callback: PythonOperator | StatefulOperator | None = None

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "scan",
            semantic_token(self.dtype),
            self.mode.value,
            self.value_kind.value,
            self.items_per_thread,
            semantic_token(self.scan_operator),
            semantic_token(self.initial_value),
            self.aggregate,
            semantic_token(self.prefix_callback),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScanSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def make_scan_semantics(
    *,
    dtype: Any,
    mode: str | ScanMode,
    value_kind: str | ScanValueKind,
    items_per_thread: int,
    scan_operator: CxxOperator | PythonOperator | StatefulOperator | None = None,
    initial_value: CxxFunction | Reference | None = None,
    aggregate: bool = False,
    prefix_callback: PythonOperator | StatefulOperator | None = None,
) -> ScanSemantics:
    """Build a scope-independent scan operation record."""

    mode = ScanMode(mode)
    value_kind = ScanValueKind(value_kind)
    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread < 1
    ):
        raise ValueError("items_per_thread must be a positive integer")
    if value_kind is ScanValueKind.SCALAR and items_per_thread != 1:
        raise ValueError("scalar scan requires items_per_thread == 1")
    if scan_operator is not None and not isinstance(scan_operator, _SCAN_OPERATORS):
        raise TypeError(f"unsupported scan operator {scan_operator!r}")
    if initial_value is not None and not isinstance(initial_value, _INITIAL_VALUES):
        raise TypeError(f"unsupported scan initial value {initial_value!r}")
    if prefix_callback is not None and not isinstance(
        prefix_callback, (PythonOperator, StatefulOperator)
    ):
        raise TypeError(f"unsupported scan prefix callback {prefix_callback!r}")
    if not isinstance(aggregate, bool):
        raise TypeError("aggregate must be a bool")
    if aggregate and prefix_callback is not None:
        raise ValueError("scan aggregate and prefix callback are mutually exclusive")

    return ScanSemantics(
        dtype=dtype,
        mode=mode,
        value_kind=value_kind,
        items_per_thread=items_per_thread,
        scan_operator=scan_operator,
        initial_value=initial_value,
        aggregate=aggregate,
        prefix_callback=prefix_callback,
    )


__all__ = [
    "ScanMode",
    "ScanSemantics",
    "ScanValueKind",
    "make_scan_semantics",
]
