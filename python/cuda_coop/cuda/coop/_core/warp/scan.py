# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe scalar CUB WarpScan calls for physical or logical warps.

A valid-prefix binding chooses a partial-scan overload. The factory supplies
an explicit plus operator and typed zero when that overload must implement
an exclusive sum. Group planning supplies membership and scratch contracts.
Runtime checks and code generation belong to the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._algorithm import Algorithm
from .._bindings import (
    ArgumentBinding,
    BindingKind,
    i32_parameter,
    normalize_i32_binding,
)
from .._types import (
    CxxFunction,
    CxxOperator,
    Dependency,
    Pointer,
    PythonOperator,
    Reference,
    TemplateParameter,
    TempStorageParameter,
)
from ..scan import ScanMode, ScanSemantics, ScanValueKind, make_scan_semantics

_SUPPORTED_LOGICAL_WARP_THREADS = frozenset({1, 2, 4, 8, 16, 32})

WarpScanMode = ScanMode


def _validate_logical_warp_threads(value: Any) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value not in _SUPPORTED_LOGICAL_WARP_THREADS
    ):
        raise ValueError(
            "threads_in_warp must be a power of two between 1 and 32"
        )
    return value


def _plus_operator() -> CxxOperator:
    return CxxOperator(
        "::cuda::std::plus<T>",
        Dependency("T"),
        name="scan_op",
    )


def _typed_zero() -> CxxFunction:
    """Spell zero in the payload type for CUB's initial-value parameter.

    The core requires a seed whose dtype is the payload type ``T``. Lowering
    replaces ``{T}`` with the bound C++ type, so the generated call passes
    ``T{0}`` instead of relying on implicit literal conversion.
    """

    return CxxFunction("{T}{0}", Dependency("T"), name="initial_value")


@dataclass(frozen=True)
class WarpScanSpecialization:
    """Keep a bound scalar WarpScan call and its logical-width controls.

    The provider describes one scalar per lane. Prefix bounds limit which
    inputs contribute; they do not change the number of lanes in the
    participating group.

    Attributes
    ----------
    specialization : Algorithm
        Bound CUB method, template arguments, parameters, and metadata.
    call : ScanSemantics
        Canonical operation, including any plus operator or zero inserted for
        a seeded or partial sum.
    threads_in_warp : int
        Logical width: one of 1, 2, 4, 8, 16, or 32.
    valid_items : ArgumentBinding
        Omitted for a full scan, static for an embedded count, or runtime for
        a count operand. Static counts are normalized and checked against the
        logical width. Runtime counts must satisfy those bounds when used.
    """

    specialization: Algorithm
    call: ScanSemantics
    threads_in_warp: int
    valid_items: ArgumentBinding

    @property
    def mode(self) -> WarpScanMode:
        return WarpScanMode(self.call.mode.value)

    @property
    def has_valid_items(self) -> bool:
        """Report whether the call uses a valid-prefix parameter.

        A count equal to the logical width still selects the partial method.
        """

        return self.valid_items.kind is not BindingKind.OMITTED

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_warp_scan_specialization(
    *,
    dtype: Any,
    threads_in_warp: int,
    mode: str | WarpScanMode,
    scan_operator: CxxOperator | PythonOperator | None = None,
    initial_value: CxxFunction | Reference | None = None,
    valid_items: bool | ArgumentBinding = False,
    warp_aggregate: bool = False,
) -> WarpScanSpecialization:
    """Bind a scalar scan and optional prefix count to a CUB WarpScan call.

    Unseeded full-group sums use CUB's sum methods. A seed or valid-prefix
    binding selects a general Scan method and inserts a plus operator when
    none was supplied. A partial exclusive sum also receives a zero in the
    payload dtype so its first valid output is defined.

    Parameters
    ----------
    dtype : object
        Input and output dtype, forwarded to ``make_scan_semantics``.
    threads_in_warp : int
        Logical width: one of 1, 2, 4, 8, 16, or 32. Booleans are rejected.
    mode : str or WarpScanMode
        ``"exclusive"`` or ``"inclusive"``.
    scan_operator : CxxOperator or PythonOperator, optional
        Static operator descriptor. ``None`` requests addition.
    initial_value : CxxFunction or Reference, optional
        Static expression or runtime scalar for an exclusive scan. Its dtype
        must match the payload or refer to ``Dependency("T")``.
    valid_items : bool or ArgumentBinding, optional
        ``False`` omits the count; ``True`` requests a runtime count operand.
        These booleans select an overload, not a numeric count. A static
        binding embeds an integer in ``[1, threads_in_warp]``. A runtime
        binding supplies the count at the call, uniformly across the group.
        This builder rejects ``None`` and plain integer counts.
    warp_aggregate : bool, optional
        Request a scalar aggregate of the valid inputs for every lane.
        The aggregate excludes the initial value.

    Returns
    -------
    WarpScanSpecialization
        Bound call with scratch first, then input and output references.
        Any seed, operator, count, and aggregate follow in CUB order.
        A runtime count uses a signed-i32 descriptor in the core signature.

    Raises
    ------
    TypeError
        A binding, static count type, operator, seed descriptor, or aggregate
        flag is invalid. Boolean static count payloads are rejected.
    ValueError
        The width, mode, dtype, seed use, or static count range is invalid.

    Notes
    -----
    The enclosing group must call the scan together, including lanes outside
    the valid prefix. Only prefixes in the valid range have defined results.
    This factory records runtime counts; it does not execute bounds checks.
    An explicitly supplied custom exclusive operator without a seed retains
    CUB's undefined first output. Group planning rejects that form.
    """

    threads_in_warp = _validate_logical_warp_threads(threads_in_warp)
    mode = WarpScanMode(mode)
    if isinstance(valid_items, bool):
        valid_items = (
            ArgumentBinding.runtime()
            if valid_items
            else ArgumentBinding.omitted()
        )
    elif not isinstance(valid_items, ArgumentBinding):
        raise TypeError("valid_items must be a bool or ArgumentBinding")
    valid_items = normalize_i32_binding(valid_items, name="valid_items")
    if valid_items.kind is BindingKind.STATIC:
        value = valid_items.value
        if not 1 <= value <= threads_in_warp:
            raise ValueError(
                "static valid_items must be between 1 and the logical warp size"
            )

    if initial_value is not None and scan_operator is None:
        scan_operator = _plus_operator()
    if (
        mode is WarpScanMode.EXCLUSIVE
        and valid_items.kind is not BindingKind.OMITTED
        and initial_value is None
        and scan_operator is None
    ):
        scan_operator = _plus_operator()
        initial_value = _typed_zero()
    elif valid_items.kind is not BindingKind.OMITTED and scan_operator is None:
        scan_operator = _plus_operator()

    call = make_scan_semantics(
        dtype=dtype,
        mode=mode,
        value_kind=ScanValueKind.SCALAR,
        items_per_thread=1,
        scan_operator=scan_operator,
        initial_value=initial_value,
        aggregate=warp_aggregate,
    )
    cpp_prefix = "Exclusive" if call.mode is ScanMode.EXCLUSIVE else "Inclusive"
    use_sum_method = (
        call.scan_operator is None
        and call.initial_value is None
        and valid_items.kind is BindingKind.OMITTED
    )
    method_name = (
        f"{cpp_prefix}Sum"
        if use_sum_method
        else f"{cpp_prefix}Scan"
        f"{'Partial' if valid_items.kind is not BindingKind.OMITTED else ''}"
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
        assert call.scan_operator is not None
        parameters.append(call.scan_operator)
    valid_items_parameter = i32_parameter(valid_items, name="valid_items")
    if valid_items_parameter is not None:
        parameters.append(valid_items_parameter)
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
        template_arguments={
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
            "valid_items": valid_items.semantic_key,
            "aggregate": call.aggregate,
            "aggregate_excludes_initial": call.aggregate,
        },
    )
    return WarpScanSpecialization(
        specialization=specialization,
        call=call,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
    )


__all__ = [
    "WarpScanMode",
    "WarpScanSpecialization",
    "make_warp_scan_specialization",
]
