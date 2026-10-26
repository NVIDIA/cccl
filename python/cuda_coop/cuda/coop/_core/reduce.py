# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe reduction inputs without choosing a group or compiler backend.

Block and Warp factories share the payload, operator, and valid-count checks
here. Their scope-specific factories add group dimensions and CUB overloads.
The resulting record also gives caches a stable description of the operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._bindings import ArgumentBinding, BindingKind, normalize_i32_binding
from ._symbols import semantic_token
from ._types import CxxOperator, PythonOperator, StatefulOperator


class ReduceOperation(str, Enum):
    """Select a custom binary reduction or the built-in sum."""

    REDUCE = "reduce"
    SUM = "sum"


class ReduceValueKind(str, Enum):
    """Distinguish one scalar per thread from a fixed per-thread array."""

    SCALAR = "scalar"
    ARRAY = "array"


_REDUCE_OPERATORS = (CxxOperator, PythonOperator, StatefulOperator)


@dataclass(frozen=True, eq=False)
class ReduceSemantics:
    """Hold normalized inputs for group, Block, and Warp reductions.

    ``value_kind`` and ``items_per_thread`` select the scalar or array form.
    ``valid_items`` records whether the call omits a count, embeds a constant,
    or accepts the count at runtime. ``reduce_operator`` is required for a
    custom reduction and absent for the built-in sum.

    Equality and hashing use semantic tokens for the dtype, binding, and
    operator. This lets separately constructed equivalent records share
    compiler artifacts without relying on their Python object identities.
    """

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
    reduce_operator: CxxOperator
    | PythonOperator
    | StatefulOperator
    | None = None,
    valid_items: bool | ArgumentBinding = False,
) -> ReduceSemantics:
    """Validate reduction inputs before selecting a Block or Warp overload.

    Scalar inputs require one item per thread. A valid-item count applies
    only to scalar inputs; its static value must be positive and fit signed
    32-bit storage. The scope-specific factory checks its group-size limit.
    For the shorthand ``valid_items``, true requests a runtime count and
    false omits it. An ``ArgumentBinding`` can instead embed a constant.

    A custom reduction needs a C++, Python, or stateful operator description.
    Sum uses its built-in operator and rejects an extra one. This function
    creates a description; a backend later turns it into callable device code.
    """

    if dtype is None:
        raise ValueError("dtype must be provided")
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
            ArgumentBinding.runtime()
            if valid_items
            else ArgumentBinding.omitted()
        )
    elif not isinstance(valid_items, ArgumentBinding):
        raise TypeError("valid_items must be a bool or ArgumentBinding")
    if valid_items.kind is not BindingKind.OMITTED:
        if value_kind is ReduceValueKind.ARRAY:
            raise ValueError("valid_items is not supported for array inputs")
        valid_items = normalize_i32_binding(valid_items, name="valid_items")
        if valid_items.kind is BindingKind.STATIC and valid_items.value < 1:
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
