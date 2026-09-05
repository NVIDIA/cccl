# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scope-independent reduction operation semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._bindings import ArgumentBinding, BindingKind, _normalize_i32_binding
from ._symbols import semantic_token
from ._types import CxxOperator, PythonOperator, StatefulOperator


class ReduceOperation(str, Enum):
    REDUCE = "reduce"
    SUM = "sum"


class ReduceValueKind(str, Enum):
    SCALAR = "scalar"
    ARRAY = "array"


_REDUCE_OPERATORS = (CxxOperator, PythonOperator, StatefulOperator)


@dataclass(frozen=True, eq=False)
class ReduceSemantics:
    dtype: Any
    operation: ReduceOperation
    value_kind: ReduceValueKind
    items_per_thread: int
    valid_items: ArgumentBinding
    reduce_operator: CxxOperator | PythonOperator | StatefulOperator | None

    @property
    def method_name(self) -> str:
        return "Reduce" if self.operation is ReduceOperation.REDUCE else "Sum"

    @property
    def has_valid_items(self) -> bool:
        return self.valid_items.kind is not BindingKind.OMITTED

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "reduce",
            semantic_token(self.dtype),
            self.operation.value,
            self.value_kind.value,
            self.items_per_thread,
            semantic_token(self.valid_items),
            semantic_token(self.reduce_operator),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReduceSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def make_reduce_semantics(
    *,
    dtype: Any,
    items_per_thread: int,
    operation: str | ReduceOperation,
    value_kind: str | ReduceValueKind,
    reduce_operator: CxxOperator | PythonOperator | StatefulOperator | None = None,
    valid_items: bool | ArgumentBinding = False,
) -> ReduceSemantics:
    """Build a scope-independent reduction operation record."""

    operation = ReduceOperation(operation)
    value_kind = ReduceValueKind(value_kind)
    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread < 1
    ):
        raise ValueError("items_per_thread must be a positive integer")
    if value_kind is ReduceValueKind.SCALAR and items_per_thread != 1:
        raise ValueError("scalar reduce requires items_per_thread == 1")
    if isinstance(valid_items, bool):
        valid_items = (
            ArgumentBinding.runtime() if valid_items else ArgumentBinding.omitted()
        )
    elif not isinstance(valid_items, ArgumentBinding):
        raise TypeError("valid_items must be a bool or ArgumentBinding")
    if (
        valid_items.kind is not BindingKind.OMITTED
        and value_kind is ReduceValueKind.ARRAY
    ):
        raise ValueError("valid_items is not supported for array inputs")
    if valid_items.kind is BindingKind.STATIC:
        valid_items = _normalize_i32_binding(valid_items, name="valid_items")
        if valid_items.value < 1:
            raise ValueError("static valid_items must be a positive integer")
    if operation is ReduceOperation.REDUCE:
        if not isinstance(reduce_operator, _REDUCE_OPERATORS):
            raise TypeError("custom reduce requires a reduce operator")
    elif reduce_operator is not None:
        raise ValueError("sum does not accept a reduce operator")

    return ReduceSemantics(
        dtype=dtype,
        operation=operation,
        value_kind=value_kind,
        items_per_thread=items_per_thread,
        valid_items=valid_items,
        reduce_operator=reduce_operator,
    )


__all__ = [
    "ReduceOperation",
    "ReduceSemantics",
    "ReduceValueKind",
    "make_reduce_semantics",
]
