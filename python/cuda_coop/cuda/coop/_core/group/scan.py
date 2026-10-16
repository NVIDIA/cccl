# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve group Scan requests into CUB calls and execution contracts.

Primitive semantics describe the values. This layer adds the resolved group,
launch dimensions, valid-prefix controls, and result ownership. Canonicalize
default choices before forming a plan so equivalent requests share identity.
The plan records required scratch and synchronization without compiling code
or allocating memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Any

from .._bindings import ArgumentBinding, BindingKind, normalize_i32_binding
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
    StatefulOperator,
)
from ..block.scan import (
    BlockScanAlgorithm,
    make_block_scan_specialization,
    normalize_block_scan_algorithm,
)
from ..launch import LaunchFacts
from ..scan import ScanMode, ScanSemantics, ScanValueKind
from ..thread_group import ThreadGroup
from ..warp.scan import make_warp_scan_specialization
from ._dispatch import _register_group_operation_family
from ._execution_requirements import (
    _build_execution_requirements,
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
    """Provide the payload-typed seed needed by a partial exclusive sum.

    Keep this descriptor in the canonical group call so planning identity and
    the later CUB signature agree on the inserted initial value.
    """

    return CxxFunction("{T}{0}", Dependency("T"), name="initial_value")


@dataclass(frozen=True, eq=False)
class GroupScanSemantics:
    """Add group-specific controls to a validated Scan operation.

    Construct ``primitive`` with ``make_scan_semantics``. This wrapper
    normalizes the optional CUB algorithm and static count representation;
    group planning checks whether those controls apply to the resolved group.
    A prefix bound limits contributing inputs while all group members still
    call the operation.

    Attributes
    ----------
    primitive : ScanSemantics
        Input shape, mode, operator, initial value, and aggregate request.
    cub_algorithm : BlockScanAlgorithm or str or None
        Optional block strategy. Construction normalizes a supplied spelling.
        Blocks default to ``RAKING``. Warps reject this option.
    valid_items : ArgumentBinding
        Omitted for all inputs, or an embedded/runtime count for a warp
        prefix. Construction normalizes static payloads to signed i32.
        Planning checks them against the logical width. Runtime counts must
        meet the same bounds and be uniform across the group.

    Notes
    -----
    Each member owns its prefix result, with the input shape and dtype. For a
    partial scan, only results in the valid prefix are defined. The aggregate
    is a separate scalar for every member and excludes the seed.
    """

    primitive: ScanSemantics
    cub_algorithm: BlockScanAlgorithm | str | None = None
    valid_items: ArgumentBinding = field(
        default_factory=ArgumentBinding.omitted
    )

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, ScanSemantics):
            raise TypeError("primitive must be ScanSemantics")
        if not isinstance(self.valid_items, ArgumentBinding):
            raise TypeError("valid_items must be an ArgumentBinding")
        object.__setattr__(
            self,
            "valid_items",
            normalize_i32_binding(self.valid_items, name="valid_items"),
        )
        if self.cub_algorithm is not None:
            try:
                algorithm = normalize_block_scan_algorithm(self.cub_algorithm)
            except ValueError as exc:
                raise ValueError(
                    "unsupported CUB BlockScan algorithm "
                    f"{self.cub_algorithm!r}"
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
    def prefix_callback(self) -> PythonOperator | StatefulOperator | None:
        return self.primitive.prefix_callback

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
    """Describe logical call arguments for planning and diagnostics.

    Record whether the operator, seed, and prefix count are static or runtime.
    The prefix callback also supplies its binding kind and role, which retain
    any runtime state operand. An aggregate buffer is an output. Mode and
    algorithm are static choices. These records describe the group call.
    The block or warp specialization factory builds CUB's parameter order,
    including its scratch and output-reference arguments.
    """

    classifications = [
        ParameterClassification(
            "value", ArgumentKind.RUNTIME, ParameterRole.INPUT
        )
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
    if operation.prefix_callback is not None:
        classifications.append(
            ParameterClassification(
                "prefix_op",
                operation.prefix_callback.argument_kind,
                operation.prefix_callback.role,
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
    """Make implicit addition explicit when the CUB overload needs it.

    Seeded exclusive addition needs a plus operator. A partial exclusive sum
    also needs a typed zero because the no-initial overload leaves rank zero
    undefined. Keep these choices in the operation used for call identity.
    The warp factory can complete normalization for other partial forms.
    """

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
    """Describe per-member prefixes and a group-wide aggregate.

    Prefixes retain the operand's scalar or array shape. The aggregate always
    has one item per member, even when each input contains several items.
    Participation constraints separately limit which partial-prefix results
    have defined values.
    """

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
    """Choose a CUB scan and record its results, participation, and scratch.

    Group dispatch has already resolved the group against exact launch
    dimensions. Canonicalize seeded sums and default algorithms before
    building the call so equivalent requests share plan identity. Reject
    custom exclusive scans without a seed or prefix callback: the public
    group result must be defined at rank zero.

    Parameters
    ----------
    call : GroupPrimitiveCall
        Requested group and operation. A new call records canonical choices
        when normalization changes the operation.
    resolved : ThreadGroup
        Block or warp group resolved against the launch dimensions.
    launch : LaunchFacts
        Exact block dimensions used to specialize CUB and count the
        independent scratch instances.
    operation : GroupScanSemantics
        Validated semantics plus the algorithm and prefix controls.

    Returns
    -------
    GroupLoweringPlan
        A CUB implementation with result and execution contracts, or an
        unsupported plan with a specific reason. Supported plans use
        implementation-owned shared scratch for each group instance and
        request a reuse barrier at the group's execution scope. The backend
        determines the scratch layout.

    Raises
    ------
    TypeError
        A scalar control or primitive descriptor has an invalid type.
    ValueError
        A static count is outside ``[1, logical_width]``, or a descriptor
        violates the selected CUB factory's validation rules.

    Notes
    -----
    Block scans support scalar and blocked-array inputs. Warp scans accept one
    scalar per lane. ``valid_items`` applies to warps, and algorithm selection
    applies to blocks. Prefix callbacks require a physical block group and
    supply an alternative to the explicit seed. Initial values and runtime
    counts must be uniform.
    Planning checks static counts. Runtime bounds remain caller preconditions;
    runtime checking belongs to the backend.
    """

    if operation.prefix_callback is not None and resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "scan prefix callbacks apply only to physical block groups",
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
    operation = _canonical_cub_scan_operation(operation)
    if operation != call.operation:
        call = GroupPrimitiveCall(group=call.group, operation=operation)
    if (
        operation.mode is GroupScanMode.EXCLUSIVE
        and operation.initial_value is None
        and operation.scan_operator is not None
        and operation.prefix_callback is None
    ):
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "group exclusive scans with a custom operator require an initial "
            "value because the no-initial overload leaves "
            "group rank zero undefined",
        )

    assert launch.exact_block_dim is not None
    block_threads = launch.exact_block_threads
    assert block_threads is not None
    if resolved.kind == "block":
        algorithm = operation.cub_algorithm or BlockScanAlgorithm.RAKING
        if (
            algorithm is BlockScanAlgorithm.WARP_SCANS
            and block_threads % 32 != 0
        ):
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "BLOCK_SCAN_WARP_SCANS requires a block size that is a "
                "multiple of the 32-thread architectural warp",
            )
        if operation.cub_algorithm is None:
            operation = replace(operation, cub_algorithm=algorithm)
            call = GroupPrimitiveCall(group=call.group, operation=operation)
        specialization = make_block_scan_specialization(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=operation.items_per_thread,
            mode=operation.mode,
            algorithm=algorithm,
            value_kind=ScanValueKind(operation.operand_kind.value),
            scan_operator=operation.scan_operator,
            initial_value=operation.initial_value,
            prefix_operator=operation.prefix_callback,
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
            if isinstance(valid_items, bool) or not isinstance(
                valid_items, Integral
            ):
                raise TypeError("static valid_items must be an integer")
            valid_items = int(valid_items)
            if not 1 <= valid_items <= warp_width:
                raise ValueError(
                    "static valid_items must be between 1 "
                    "and the logical warp size"
                )
        warp_specialization = make_warp_scan_specialization(
            dtype=operation.dtype,
            threads_in_warp=warp_width,
            mode=operation.mode,
            scan_operator=operation.scan_operator,
            initial_value=operation.initial_value,
            valid_items=operation.valid_items,
            warp_aggregate=operation.aggregate,
        )
        canonical_primitive = warp_specialization.call
        if canonical_primitive != operation.primitive:
            operation = replace(operation, primitive=canonical_primitive)
            call = GroupPrimitiveCall(group=call.group, operation=operation)
        specialization = warp_specialization.specialization
        target = GroupLoweringTarget.CUB_WARP
        cpp_class = "cub::WarpScan"
        header = "cub/warp/warp_scan.cuh"

    result = _result_contract(operation)
    requirements = _build_execution_requirements(
        resolved,
        launch,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=(
            *(
                ("initial_value",)
                if operation.initial_value is not None
                else ()
            ),
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
        implementation=specialization,
        topology=requirements.topology,
        participation=requirements.participation,
        result=result,
        synchronization=requirements.synchronization,
        temp_storage=requirements.temp_storage,
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=cpp_class,
            method=specialization.method_name,
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


__all__ = [
    "GroupScanMode",
    "GroupScanSemantics",
]
