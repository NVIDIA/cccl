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
    Dependency,
    PythonOperator,
    Reference,
    StatefulOperator,
)


class ScanMode(str, Enum):
    """Whether each output includes its corresponding input."""

    EXCLUSIVE = "exclusive"
    INCLUSIVE = "inclusive"


class ScanValueKind(str, Enum):
    """Per-thread operand form presented to the collective."""

    SCALAR = "scalar"
    ARRAY = "array"


_SCAN_OPERATORS = (CxxOperator, PythonOperator)
_INITIAL_VALUES = (CxxFunction, Reference)
_SCAN_OPERATOR_ALIASES = {
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


def normalize_scan_operator_alias(value: object) -> str | None:
    """Normalize one string alias shared by every public Scan spelling."""

    if not isinstance(value, str):
        raise TypeError("scan_op must be a string")
    token = value.strip().lower().replace("-", "_")
    return _SCAN_OPERATOR_ALIASES.get(token)


_PREFIX_CALLBACKS = (PythonOperator, StatefulOperator)


def _initial_dtype_matches(dtype: Any, initial_value: CxxFunction | Reference) -> bool:
    initial_dtype = initial_value.dtype
    if isinstance(initial_dtype, Dependency):
        return initial_dtype.name == "T"
    return semantic_token(initial_dtype) == semantic_token(dtype)


@dataclass(frozen=True, eq=False)
class ScanSemantics:
    """Normalized scan payload, operator, and result contract."""

    dtype: Any
    mode: ScanMode
    value_kind: ScanValueKind
    items_per_thread: int
    scan_operator: CxxOperator | PythonOperator | None = None
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
    scan_operator: CxxOperator | PythonOperator | None = None,
    initial_value: CxxFunction | Reference | None = None,
    aggregate: bool = False,
    prefix_callback: PythonOperator | StatefulOperator | None = None,
) -> ScanSemantics:
    """Build a scope-independent scan operation record."""

    if dtype is None:
        raise ValueError("dtype must be provided")
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
    if initial_value is not None:
        if not isinstance(initial_value, _INITIAL_VALUES):
            raise TypeError(f"unsupported scan initial value {initial_value!r}")
        if not _initial_dtype_matches(dtype, initial_value):
            raise TypeError(
                "scan initial_value dtype must exactly match the payload dtype"
            )
        if mode is ScanMode.INCLUSIVE:
            raise ValueError("inclusive scans do not accept an initial value")
    if not isinstance(aggregate, bool):
        raise TypeError("aggregate must be a bool")
    if prefix_callback is not None and not isinstance(
        prefix_callback, _PREFIX_CALLBACKS
    ):
        raise TypeError(f"unsupported scan prefix callback {prefix_callback!r}")
    if initial_value is not None and prefix_callback is not None:
        raise ValueError(
            "scan initial value and prefix callback are mutually exclusive"
        )
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
    "normalize_scan_operator_alias",
]
