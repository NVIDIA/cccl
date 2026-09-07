# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB WarpReduce semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._bindings import ArgumentBinding, BindingKind, _normalize_i32_binding
from .._symbols import semantic_token
from .._types import (
    INT32,
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
from ..reduce import ReduceSemantics, make_reduce_semantics
from ._common import _validate_logical_warp_threads


class WarpReduceOperation(str, Enum):
    """CUB WarpReduce entry point selected by a frontend."""

    REDUCE = "reduce"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


_REDUCE_OPERATORS = (CxxOperator, PythonOperator, StatefulOperator)


@dataclass(frozen=True)
class WarpReduceSpec:
    """Fully specialized WarpReduce call semantics."""

    specialization: AlgorithmSpec
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


def make_warp_reduce_spec(
    *,
    dtype: Any,
    threads_in_warp: int,
    operation: str | WarpReduceOperation,
    reduce_operator: CxxOperator | PythonOperator | StatefulOperator | None = None,
    valid_items: bool | ArgumentBinding = False,
    include_full_warp: bool = False,
) -> WarpReduceSpec:
    """Build canonical scalar WarpReduce semantics."""

    operation = WarpReduceOperation(operation)
    threads_in_warp = _validate_logical_warp_threads(threads_in_warp)
    if isinstance(valid_items, bool):
        valid_items = (
            ArgumentBinding.runtime() if valid_items else ArgumentBinding.omitted()
        )
    elif not isinstance(valid_items, ArgumentBinding):
        raise TypeError("valid_items must be a bool or ArgumentBinding")
    if valid_items.kind is BindingKind.STATIC:
        valid_items = _normalize_i32_binding(valid_items, name="valid_items")
        value = valid_items.value
        if value < 1:
            raise ValueError("static valid_items must be a positive integer")
        if value > threads_in_warp:
            raise ValueError(
                f"static valid_items {value} exceeds warp size {threads_in_warp}"
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
        items_per_thread=1,
        operation=("sum" if operation is WarpReduceOperation.SUM else "reduce"),
        value_kind="scalar",
        reduce_operator=canonical_operator,
        valid_items=valid_items,
    )
    base_parameters: list[Any] = [
        TempStorageParameter(),
        Reference(Dependency("T"), name="input"),
    ]
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
        methods.append((*base_parameters, Value(INT32, name="valid_items"), output))
    elif valid_items.kind is BindingKind.STATIC:
        methods.append(
            (
                *base_parameters,
                CxxFunction(str(valid_items.value), INT32, name="valid_items"),
                output,
            )
        )

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
    ).specialize(
        {
            "T": dtype,
            "VIRTUAL_WARP_THREADS": threads_in_warp,
        },
        metadata={
            "scope": "warp",
            "primitive": "reduce",
            "operation": operation,
            "operator": (
                None if reduce_operator is None else type(reduce_operator).__qualname__
            ),
            "valid_items": semantic_token(valid_items),
            "full_warp": (valid_items.kind is BindingKind.OMITTED or include_full_warp),
        },
    )
    return WarpReduceSpec(
        specialization=specialization,
        call=call,
        operation=operation,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        has_full_warp=(valid_items.kind is BindingKind.OMITTED or include_full_warp),
    )
