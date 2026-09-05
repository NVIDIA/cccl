# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scan semantics and lowering for explicit thread groups.

The planner preserves portable scan meaning while selecting CUDAX or the
matching CUB block or warp specialization. It does not own backend activation,
compiler rewrites, or provider rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._bindings import ArgumentBinding, BindingKind
from .._symbols import semantic_token
from .._types import (
    CxxFunction,
    CxxOperator,
    PythonOperator,
    Reference,
    StatefulOperator,
)
from ..block.scan import (
    BlockScanAlgorithm,
    ScanMode,
    ScanValueKind,
    make_block_scan_spec,
    normalize_block_scan_algorithm,
)
from ..launch import LaunchFacts
from ..scan import ScanSemantics
from ..thread_group import ThreadGroup
from ..warp.scan import WarpScanMode, make_warp_scan_spec
from ._contracts import (
    _contracts,
    _stateful_operator_uniformity,
    _unsupported,
    _unsupported_cub_warp_width,
)
from ._model import (
    ArgumentPrecondition,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupPrimitiveCall,
    ImplementationProvenance,
    PreconditionEnforcement,
    ResultVisibility,
    StorageOwnership,
    UnsupportedReasonCode,
)

GroupScanMode = ScanMode


@dataclass(frozen=True, eq=False)
class GroupScanSemantics:
    primitive: ScanSemantics
    cub_algorithm: BlockScanAlgorithm | str | None = None
    valid_items: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, ScanSemantics):
            raise TypeError("primitive must be ScanSemantics")
        if not isinstance(self.valid_items, ArgumentBinding):
            raise TypeError("valid_items must be an ArgumentBinding")
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
    def scan_operator(self) -> CxxOperator | PythonOperator | StatefulOperator | None:
        return self.primitive.scan_operator

    @property
    def initial_value(self) -> CxxFunction | Reference | None:
        return self.primitive.initial_value

    @property
    def aggregate(self) -> bool:
        return self.primitive.aggregate

    @property
    def prefix_callback(self) -> PythonOperator | StatefulOperator | None:
        return self.primitive.prefix_callback

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


def _plan_scan(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupScanSemantics,
) -> GroupLoweringPlan:
    if resolved.kind not in {"block", "warp", "threads_within_warp"}:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group scan supports physical block, physical-warp, and "
            "logical-warp groups",
        )
    if operation.prefix_callback is not None:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "group scan prefix callbacks are not supported in the initial slice",
        )
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
    if operation.initial_value is not None and operation.scan_operator is None:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "CUB sum scan overloads do not accept an explicit initial value; "
            "provide a scan operator",
        )
    if (
        operation.mode is GroupScanMode.EXCLUSIVE
        and operation.scan_operator is not None
        and operation.initial_value is None
    ):
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "group exclusive scans with an explicit operator require an initial "
            "value because the CUB no-initial overload leaves group rank zero "
            "undefined",
        )
    assert launch.exact_block_dim is not None
    block_threads = launch.exact_block_threads
    assert block_threads is not None
    if resolved.kind in {"warp", "threads_within_warp"}:
        warp_width, width_error = _unsupported_cub_warp_width(call, resolved)
        if width_error is not None:
            return width_error
        assert warp_width is not None
        if operation.valid_items.kind is BindingKind.STATIC:
            valid_items = operation.valid_items.value
            if isinstance(valid_items, bool) or not isinstance(valid_items, int):
                raise TypeError("static valid_items must be an integer")
            if not 1 <= valid_items <= warp_width:
                raise ValueError(
                    "static valid_items must be between 1 and the logical warp size"
                )
        if operation.operand_kind is GroupOperandKind.ARRAY:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERAND_FORM,
                "CUB WarpScan is scalar-per-lane; multi-item warp scan is unsupported",
            )
        if operation.cub_algorithm is not None:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "CUB algorithm selection applies to BlockScan, not WarpScan",
            )
        spec = make_warp_scan_spec(
            dtype=operation.dtype,
            threads_in_warp=warp_width,
            mode=WarpScanMode(operation.mode.value),
            scan_operator=operation.scan_operator,
            initial_value=operation.initial_value,
            warp_aggregate=operation.aggregate,
            valid_items=operation.valid_items.kind is not BindingKind.OMITTED,
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        cpp_class = "cub::WarpScan"
        header = "cub/warp/warp_scan.cuh"
    else:
        if (
            operation.cub_algorithm is BlockScanAlgorithm.WARP_SCANS
            and block_threads % 32 != 0
        ):
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "BLOCK_SCAN_WARP_SCANS requires a block size that is a multiple "
                "of the 32-thread architectural warp; CUB otherwise substitutes "
                "BLOCK_SCAN_RAKING",
            )
        if (
            operation.mode is GroupScanMode.INCLUSIVE
            and operation.operand_kind is GroupOperandKind.SCALAR
            and operation.initial_value is not None
        ):
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "scalar CUB BlockScan InclusiveScan has no initial-value overload",
            )
        spec = make_block_scan_spec(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=operation.items_per_thread,
            mode=ScanMode(operation.mode.value),
            algorithm=operation.cub_algorithm or BlockScanAlgorithm.RAKING,
            value_kind=ScanValueKind(operation.operand_kind.value),
            scan_operator=operation.scan_operator,
            initial_value=operation.initial_value,
            block_aggregate=operation.aggregate,
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        cpp_class = "cub::BlockScan"
        header = "cub/block/block_scan.cuh"
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=(
            *_stateful_operator_uniformity(operation.scan_operator),
            *(("initial_value",) if operation.initial_value is not None else ()),
            *(
                ("valid_items",)
                if operation.valid_items.kind is not BindingKind.OMITTED
                else ()
            ),
        ),
        valid_member_selection=(
            "first valid_items lanes"
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
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=cpp_class,
            method=spec.method_name,
        ),
    )


__all__ = ["GroupScanMode", "GroupScanSemantics"]
