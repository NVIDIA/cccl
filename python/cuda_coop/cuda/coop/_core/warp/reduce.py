# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe CUB reductions for physical or fixed-width logical warps.

The factory selects a named method and its full-warp or valid-count overloads.
It records Min and Max in ``call`` as equivalent reductions with C++ operators
for semantic comparison. The bound algorithm still calls the named CUB method.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm
from .._bindings import ArgumentBinding, BindingKind, normalize_i32_binding
from .._symbols import semantic_token
from .._types import (
    INT32,
    Array,
    CxxFunction,
    CxxOperator,
    Dependency,
    PythonOperator,
    Reference,
    StatefulOperator,
    TemplateParameter,
    TempStorageParameter,
    Value,
)
from ..reduce import ReduceSemantics, ReduceValueKind, make_reduce_semantics


def _validate_logical_warp_threads(value: Any) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= 32
    ):
        raise ValueError("threads_in_warp must be an integer between 1 and 32")
    return value


class WarpReduceOperation(str, Enum):
    """CUB WarpReduce entry point selected by a frontend."""

    REDUCE = "reduce"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


_REDUCE_OPERATORS = (CxxOperator, PythonOperator, StatefulOperator)


@dataclass(frozen=True)
class WarpReduceSpecialization:
    """Keep a bound WarpReduce algorithm and its supported count signatures.

    ``operation`` retains the named CUB method. The shared ``call`` represents
    Min and Max as reductions with C++ operators for semantic comparison.
    ``has_full_warp`` records whether a signature without a valid count is
    available; it does not prove which threads participate in a kernel call.
    """

    specialization: Algorithm
    call: ReduceSemantics
    operation: WarpReduceOperation
    threads_in_warp: int
    valid_items: ArgumentBinding
    has_full_warp: bool

    @property
    def has_valid_items(self) -> bool:
        return self.valid_items.kind is not BindingKind.OMITTED

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_warp_reduce_specialization(
    *,
    dtype: Any,
    threads_in_warp: int,
    operation: str | WarpReduceOperation,
    items_per_thread: int = 1,
    value_kind: str | ReduceValueKind = ReduceValueKind.SCALAR,
    reduce_operator: CxxOperator
    | PythonOperator
    | StatefulOperator
    | None = None,
    valid_items: bool | ArgumentBinding = False,
    include_full_warp: bool = False,
) -> WarpReduceSpecialization:
    """Bind a scalar or fixed-array reduction for a logical warp.

    The width must be an integer from 1 through 32. A static valid count
    must fit within that width. Min and Max accept no valid count or custom
    operator; Reduce requires an operator description and Sum uses its own.
    Array inputs contribute every item from every lane. Valid counts apply
    only to scalar inputs.

    ``valid_items=True`` selects a runtime count and false omits it. An
    ``ArgumentBinding`` can embed a constant instead. ``include_full_warp``
    adds a no-count signature alongside a requested count signature, so one
    bound algorithm can expose both forms. It requires a valid-count request.

    The returned record describes CUB parameters and template arguments.
    The backend later allocates scratch and emits code for those signatures.
    """

    operation = WarpReduceOperation(operation)
    threads_in_warp = _validate_logical_warp_threads(threads_in_warp)
    if isinstance(valid_items, bool):
        valid_items = (
            ArgumentBinding.runtime()
            if valid_items
            else ArgumentBinding.omitted()
        )
    elif not isinstance(valid_items, ArgumentBinding):
        raise TypeError("valid_items must be a bool or ArgumentBinding")
    if valid_items.kind is BindingKind.STATIC:
        valid_items = normalize_i32_binding(valid_items, name="valid_items")
        value = valid_items.value
        if value < 1:
            raise ValueError("static valid_items must be a positive integer")
        if value > threads_in_warp:
            raise ValueError(
                f"static valid_items {value} exceeds warp size "
                f"{threads_in_warp}"
            )
    if operation in {WarpReduceOperation.MIN, WarpReduceOperation.MAX} and (
        valid_items.kind is not BindingKind.OMITTED
    ):
        raise ValueError(
            f"WarpReduce {operation.value} does not accept valid_items"
        )
    if operation is WarpReduceOperation.REDUCE:
        if not isinstance(reduce_operator, _REDUCE_OPERATORS):
            raise TypeError("custom WarpReduce requires a reduce operator")
    elif reduce_operator is not None:
        raise ValueError(f"{operation.value} does not accept a reduce operator")
    if include_full_warp and valid_items.kind is BindingKind.OMITTED:
        raise ValueError("include_full_warp requires a valid_items signature")

    method_name = {
        WarpReduceOperation.REDUCE: "Reduce",
        WarpReduceOperation.SUM: "Sum",
        WarpReduceOperation.MIN: "Min",
        WarpReduceOperation.MAX: "Max",
    }[operation]
    canonical_operator = reduce_operator
    if operation in {WarpReduceOperation.MIN, WarpReduceOperation.MAX}:
        canonical_operator = CxxOperator(
            (
                "::cuda::minimum<>"
                if operation is WarpReduceOperation.MIN
                else "::cuda::maximum<>"
            ),
            dtype,
        )
    call = make_reduce_semantics(
        dtype=dtype,
        items_per_thread=items_per_thread,
        operation=("sum" if operation is WarpReduceOperation.SUM else "reduce"),
        value_kind=value_kind,
        reduce_operator=canonical_operator,
        valid_items=valid_items,
    )
    base_parameters: list[Any] = [TempStorageParameter()]
    if call.value_kind is ReduceValueKind.ARRAY:
        base_parameters.append(
            Array(
                Dependency("T"),
                Dependency("ITEMS_PER_THREAD"),
                name="input",
            )
        )
    else:
        base_parameters.append(Reference(Dependency("T"), name="input"))
    if reduce_operator is not None:
        base_parameters.append(reduce_operator)
    output = Reference(
        Dependency("T"),
        name="output",
        is_output=True,
        is_return=True,
    )
    methods: list[tuple[Any, ...]] = []
    if valid_items.kind is BindingKind.OMITTED or include_full_warp:
        methods.append((*base_parameters, output))
    if valid_items.kind is BindingKind.RUNTIME:
        methods.append(
            (*base_parameters, Value(INT32, name="valid_items"), output)
        )
    elif valid_items.kind is BindingKind.STATIC:
        methods.append(
            (
                *base_parameters,
                CxxFunction(str(valid_items.value), INT32, name="valid_items"),
                output,
            )
        )

    template_arguments = {
        "T": dtype,
        "VIRTUAL_WARP_THREADS": threads_in_warp,
    }
    if call.value_kind is ReduceValueKind.ARRAY:
        template_arguments["ITEMS_PER_THREAD"] = items_per_thread

    specialization = Algorithm(
        struct_name="WarpReduce",
        method_name=method_name,
        c_name="warp_reduce",
        includes=("cub/warp/warp_reduce.cuh",),
        template_parameters=(
            TemplateParameter("T"),
            TemplateParameter("VIRTUAL_WARP_THREADS"),
        ),
        parameters=tuple(methods),
        template_arguments=template_arguments,
        metadata={
            "scope": "warp",
            "primitive": "reduce",
            "operation": operation,
            "value_kind": call.value_kind,
            "operator": (
                None
                if reduce_operator is None
                else type(reduce_operator).__qualname__
            ),
            "valid_items": semantic_token(valid_items),
            "full_warp": (
                valid_items.kind is BindingKind.OMITTED or include_full_warp
            ),
        },
    )
    return WarpReduceSpecialization(
        specialization=specialization,
        call=call,
        operation=operation,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        has_full_warp=(
            valid_items.kind is BindingKind.OMITTED or include_full_warp
        ),
    )


__all__ = [
    "WarpReduceOperation",
    "WarpReduceSpecialization",
    "_validate_logical_warp_threads",
    "make_warp_reduce_specialization",
]
