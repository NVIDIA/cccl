# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB WarpScan semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._types import (
    INT32,
    CxxFunction,
    CxxOperator,
    Dependency,
    Pointer,
    PythonOperator,
    Reference,
    StatefulOperator,
    TemplateParameter,
    TempStorageParameter,
    Value,
)
from ..scan import (
    ScanMode,
    ScanSemantics,
    ScanValueKind,
    make_scan_semantics,
)
from ._common import _validate_logical_warp_threads

WarpScanMode = ScanMode


@dataclass(frozen=True)
class WarpScanSpec:
    """Fully specialized scalar WarpScan call semantics."""

    specialization: AlgorithmSpec
    call: ScanSemantics
    mode: WarpScanMode
    threads_in_warp: int
    has_initial_value: bool
    has_valid_items: bool
    has_warp_aggregate: bool

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def uses_sum_method(self) -> bool:
        return self.specialization.metadata["operator"] is None

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_warp_scan_spec(
    *,
    dtype: Any,
    threads_in_warp: int,
    mode: str | WarpScanMode,
    scan_operator: CxxOperator | PythonOperator | StatefulOperator | None = None,
    initial_value: CxxFunction | Reference | None = None,
    valid_items: bool = False,
    warp_aggregate: bool = False,
) -> WarpScanSpec:
    """Build canonical scalar WarpScan semantics."""

    threads_in_warp = _validate_logical_warp_threads(threads_in_warp)
    if not isinstance(valid_items, bool):
        raise TypeError("valid_items must be a bool")
    if valid_items and scan_operator is None:
        scan_operator = CxxOperator(
            "::cuda::std::plus<T>",
            Dependency("T"),
            name="scan_op",
        )
    call = make_scan_semantics(
        dtype=dtype,
        mode=mode,
        value_kind=ScanValueKind.SCALAR,
        items_per_thread=1,
        scan_operator=scan_operator,
        initial_value=initial_value,
        aggregate=warp_aggregate,
    )

    cpp_prefix = "Exclusive" if call.mode is WarpScanMode.EXCLUSIVE else "Inclusive"
    use_sum_method = (
        call.scan_operator is None and call.initial_value is None and not valid_items
    )
    method_name = (
        f"{cpp_prefix}Sum"
        if use_sum_method
        else f"{cpp_prefix}Scan{'Partial' if valid_items else ''}"
    )

    parameters: list[Any] = [
        TempStorageParameter(),
        Reference(Dependency("T"), name="input"),
        Reference(
            Dependency("T"),
            name="output",
            is_output=True,
            is_return=True,
        ),
    ]
    if call.initial_value is not None:
        parameters.append(call.initial_value)
    if not use_sum_method:
        if call.scan_operator is None:
            raise ValueError("non-default WarpScan requires a scan operator")
        parameters.append(call.scan_operator)
    if valid_items:
        parameters.append(Value(INT32, name="valid_items"))
    if call.aggregate:
        parameters.append(
            Pointer(
                Dependency("T"),
                name="warp_aggregate",
                is_output=True,
                is_return=False,
                is_array_pointer=True,
                deref_on_call=True,
            )
        )

    specialization = Algorithm(
        struct_name="WarpScan",
        method_name=method_name,
        c_name="warp_scan",
        includes=("cub/warp/warp_scan.cuh",),
        template_parameters=(
            TemplateParameter("T"),
            TemplateParameter("VIRTUAL_WARP_THREADS"),
        ),
        parameters=(tuple(parameters),),
        fake_return=True,
    ).specialize(
        {
            "T": dtype,
            "VIRTUAL_WARP_THREADS": threads_in_warp,
        },
        metadata={
            "scope": "warp",
            "primitive": "scan",
            "mode": call.mode,
            "operator": (
                None
                if call.scan_operator is None
                else type(call.scan_operator).__qualname__
            ),
            "initial_value": call.initial_value is not None,
            "valid_items": valid_items,
            "warp_aggregate": call.aggregate,
        },
    )
    return WarpScanSpec(
        specialization=specialization,
        call=call,
        mode=call.mode,
        threads_in_warp=threads_in_warp,
        has_initial_value=call.initial_value is not None,
        has_valid_items=valid_items,
        has_warp_aggregate=call.aggregate,
    )
