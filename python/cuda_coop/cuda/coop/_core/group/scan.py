# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scan semantics for physical block and physical or logical warp groups."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Any

from .._bindings import ArgumentBinding, BindingKind, _normalize_i32_binding
from .._symbols import semantic_token
from .._types import (
    ArgumentKind,
    CxxFunction,
    CxxOperator,
    Dependency,
    ParameterClassification,
    ParameterRole,
    PythonOperator,
    Reference,
)
from ..block.scan import (
    BlockScanAlgorithm,
    make_block_scan_spec,
    normalize_block_scan_algorithm,
)
from ..launch import LaunchFacts
from ..scan import ScanMode, ScanSemantics, ScanValueKind
from ..thread_group import ThreadGroup
from ..warp.scan import make_warp_scan_spec
from ._contracts import _contracts, _unsupported, _unsupported_cub_warp_width
from ._dispatch import _register_group_operation_family
from ._model import (
    ArgumentPrecondition,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupPrimitiveCall,
    ImplementationProvenance,
    LogicalResultContract,
    PreconditionEnforcement,
    ResultContract,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    UnsupportedReasonCode,
)

GroupScanMode = ScanMode


def _plus_operator() -> CxxOperator:
    return CxxOperator(
        "::cuda::std::plus<T>",
        Dependency("T"),
        name="scan_op",
    )


def _typed_zero() -> CxxFunction:
    return CxxFunction("{T}{0}", Dependency("T"), name="initial_value")


@dataclass(frozen=True, eq=False)
class GroupScanSemantics:
    """Out-of-place CUB scan semantics selected after group resolution."""

    primitive: ScanSemantics
    cub_algorithm: BlockScanAlgorithm | str | None = None
    valid_items: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, ScanSemantics):
            raise TypeError("primitive must be ScanSemantics")
        if not isinstance(self.valid_items, ArgumentBinding):
            raise TypeError("valid_items must be an ArgumentBinding")
        object.__setattr__(
            self,
            "valid_items",
            _normalize_i32_binding(self.valid_items, name="valid_items"),
        )
        if self.cub_algorithm is not None:
            try:
                algorithm = normalize_block_scan_algorithm(self.cub_algorithm)
            except ValueError as exc:
                raise ValueError(
                    f"unsupported CUB BlockScan algorithm {self.cub_algorithm!r}"
                ) from exc
            object.__setattr__(self, "cub_algorithm", algorithm)

    @property
    def dtype(self) -> Any:
        return self.primitive.dtype

    @property
    def mode(self) -> GroupScanMode:
        return GroupScanMode(self.primitive.mode.value)

    @property
    def operand_kind(self) -> GroupOperandKind:
        return GroupOperandKind(self.primitive.value_kind.value)

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def scan_operator(self) -> CxxOperator | PythonOperator | None:
        return self.primitive.scan_operator

    @property
    def initial_value(self) -> CxxFunction | Reference | None:
        return self.primitive.initial_value

    @property
    def aggregate(self) -> bool:
        return self.primitive.aggregate

    @property
    def result_visibility(self) -> ResultVisibility:
        return ResultVisibility.PER_MEMBER

    @property
    def returns_value(self) -> bool:
        return True

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.primitive.semantic_key,
            None if self.cub_algorithm is None else self.cub_algorithm.value,
            (
                self.valid_items.kind.value,
                semantic_token(self.valid_items.value),
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupScanSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _call_classifications(
    operation: GroupScanSemantics,
) -> tuple[ParameterClassification, ...]:
    classifications = [
        ParameterClassification("value", ArgumentKind.RUNTIME, ParameterRole.INPUT)
    ]
    if operation.scan_operator is not None:
        classifications.append(
            ParameterClassification(
                "scan_op",
                operation.scan_operator.argument_kind,
                operation.scan_operator.role,
            )
        )
    if operation.initial_value is not None:
        classifications.append(
            ParameterClassification(
                "initial_value",
                operation.initial_value.argument_kind,
                operation.initial_value.role,
            )
        )
    if operation.valid_items.kind is not BindingKind.OMITTED:
        argument_kind = operation.valid_items.argument_kind
        assert argument_kind is not None
        classifications.append(
            ParameterClassification(
                "valid_items",
                argument_kind,
                (
                    ParameterRole.CONSTANT
                    if argument_kind is ArgumentKind.STATIC
                    else ParameterRole.INPUT
                ),
            )
        )
    if operation.aggregate:
        classifications.append(
            ParameterClassification(
                "aggregate_output",
                ArgumentKind.RUNTIME,
                ParameterRole.OUTPUT,
            )
        )
    classifications.extend(
        (
            ParameterClassification(
                "mode",
                ArgumentKind.STATIC,
                ParameterRole.CONSTANT,
            ),
            ParameterClassification(
                "algorithm",
                ArgumentKind.STATIC,
                ParameterRole.CONSTANT,
            ),
        )
    )
    return tuple(classifications)


def _canonical_cub_scan_operation(
    operation: GroupScanSemantics,
) -> GroupScanSemantics:
    primitive = operation.primitive
    if (
        operation.mode is GroupScanMode.EXCLUSIVE
        and operation.initial_value is not None
        and operation.scan_operator is None
    ):
        primitive = replace(primitive, scan_operator=_plus_operator())
    if (
        operation.mode is GroupScanMode.EXCLUSIVE
        and operation.valid_items.kind is not BindingKind.OMITTED
        and operation.initial_value is None
        and operation.scan_operator is None
    ):
        primitive = replace(
            primitive,
            scan_operator=_plus_operator(),
            initial_value=_typed_zero(),
        )
    return replace(operation, primitive=primitive)


def _result_contract(operation: GroupScanSemantics) -> ResultContract:
    results = [
        LogicalResultContract(
            name="value",
            dtype=operation.dtype,
            visibility=ResultVisibility.PER_MEMBER,
            ownership=ResultOwnership.EACH_MEMBER,
            operand_kind=operation.operand_kind,
            items_per_member=operation.items_per_thread,
        )
    ]
    if operation.aggregate:
        results.append(
            LogicalResultContract(
                name="aggregate",
                dtype=operation.dtype,
                visibility=ResultVisibility.ALL_MEMBERS,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.SCALAR,
                items_per_member=1,
            )
        )
    return ResultContract(tuple(results))


def _plan_scan(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupScanSemantics,
) -> GroupLoweringPlan:
    if (
        operation.valid_items.kind is not BindingKind.OMITTED
        and resolved.kind == "block"
    ):
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "valid_items applies to WarpScan, not BlockScan",
        )
    operation = _canonical_cub_scan_operation(operation)
    if operation != call.operation:
        call = GroupPrimitiveCall(group=call.group, operation=operation)
    if (
        operation.mode is GroupScanMode.EXCLUSIVE
        and operation.initial_value is None
        and operation.scan_operator is not None
    ):
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "group exclusive scans with a custom operator require an initial "
            "value because the no-initial overload leaves group rank zero undefined",
        )

    assert launch.exact_block_dim is not None
    block_threads = launch.exact_block_threads
    assert block_threads is not None
    if resolved.kind == "block":
        algorithm = operation.cub_algorithm or BlockScanAlgorithm.RAKING
        if algorithm is BlockScanAlgorithm.WARP_SCANS and block_threads % 32 != 0:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "BLOCK_SCAN_WARP_SCANS requires a block size that is a multiple "
                "of the 32-thread architectural warp",
            )
        if operation.cub_algorithm is None:
            operation = replace(operation, cub_algorithm=algorithm)
            call = GroupPrimitiveCall(group=call.group, operation=operation)
        spec = make_block_scan_spec(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=operation.items_per_thread,
            mode=operation.mode,
            algorithm=algorithm,
            value_kind=ScanValueKind(operation.operand_kind.value),
            scan_operator=operation.scan_operator,
            initial_value=operation.initial_value,
            block_aggregate=operation.aggregate,
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        cpp_class = "cub::BlockScan"
        header = "cub/block/block_scan.cuh"
    else:
        if operation.cub_algorithm is not None:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "CUB algorithm selection applies to BlockScan, not WarpScan",
            )
        if operation.operand_kind is GroupOperandKind.ARRAY:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERAND_FORM,
                "CUB WarpScan supports one scalar value per lane",
            )
        warp_width, width_error = _unsupported_cub_warp_width(call, resolved)
        if width_error is not None:
            return width_error
        assert warp_width is not None
        if operation.valid_items.kind is BindingKind.STATIC:
            valid_items = operation.valid_items.value
            if isinstance(valid_items, bool) or not isinstance(valid_items, Integral):
                raise TypeError("static valid_items must be an integer")
            valid_items = int(valid_items)
            if not 1 <= valid_items <= warp_width:
                raise ValueError(
                    "static valid_items must be between 1 and the logical warp size"
                )
        warp_spec = make_warp_scan_spec(
            dtype=operation.dtype,
            threads_in_warp=warp_width,
            mode=operation.mode,
            scan_operator=operation.scan_operator,
            initial_value=operation.initial_value,
            valid_items=operation.valid_items,
            warp_aggregate=operation.aggregate,
        )
        canonical_primitive = warp_spec.call
        if canonical_primitive != operation.primitive:
            operation = replace(operation, primitive=canonical_primitive)
            call = GroupPrimitiveCall(group=call.group, operation=operation)
        spec = warp_spec.specialization
        target = GroupLoweringTarget.CUB_WARP
        cpp_class = "cub::WarpScan"
        header = "cub/warp/warp_scan.cuh"

    result = _result_contract(operation)
    contracts = _contracts(
        resolved,
        launch,
        result=result,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=(
            *(("initial_value",) if operation.initial_value is not None else ()),
            *(
                ("valid_items",)
                if operation.valid_items.kind is not BindingKind.OMITTED
                else ()
            ),
        ),
        valid_member_selection=(
            "first valid_items lanes by linear group rank"
            if operation.valid_items.kind is not BindingKind.OMITTED
            else None
        ),
        argument_preconditions=(
            (
                ArgumentPrecondition(
                    name="valid_items",
                    minimum=1,
                    maximum=resolved.static_size,
                    enforcement=(
                        PreconditionEnforcement.PLANNER_VALIDATED
                        if operation.valid_items.kind is BindingKind.STATIC
                        else PreconditionEnforcement.CALLER
                    ),
                ),
            )
            if operation.valid_items.kind is not BindingKind.OMITTED
            else ()
        ),
    )
    return GroupLoweringPlan(
        target=target,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        topology=contracts[0],
        participation=contracts[1],
        result=result,
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=cpp_class,
            method=spec.method_name,
        ),
    )


_register_group_operation_family(
    GroupScanSemantics,
    classifications=_call_classifications,
    planner=_plan_scan,
    group_kinds=frozenset({"block", "warp", "threads_within_warp"}),
    unsupported_group_message=(
        "cuda.coop Scan supports this_block(), complete physical this_warp(), "
        "and power-of-two logical-warp groups"
    ),
)


__all__ = ["GroupScanMode", "GroupScanSemantics"]
