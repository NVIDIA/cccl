# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan CUB block and warp reductions with leader-owned results.

The planner validates each group's CUB operand and algorithm constraints and
carries explicit or implementation-owned scratch requirements to the backend.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

from .._bindings import ArgumentBinding, BindingKind
from .._types import (
    ArgumentKind,
    CxxOperator,
    ParameterClassification,
    ParameterRole,
    PythonOperator,
    StatefulOperator,
)
from ..block.reduce import (
    BlockReduceAlgorithm,
    make_block_reduce_specialization,
    normalize_block_reduce_algorithm,
)
from ..launch import LaunchFacts
from ..reduce import ReduceOperation, ReduceSemantics, ReduceValueKind
from ..thread_group import ThreadGroup
from ..warp.reduce import WarpReduceOperation, make_warp_reduce_specialization
from ._dispatch import _register_group_operation_family
from ._execution_requirements import (
    _build_execution_requirements,
    _unsupported,
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

_COMMUTATIVE_REDUCE_OPERATORS = frozenset(
    {
        "::cuda::std::plus<>",
        "::cuda::std::multiplies<>",
        "::cuda::minimum<>",
        "::cuda::maximum<>",
        "::cuda::std::bit_and<>",
        "::cuda::std::bit_or<>",
        "::cuda::std::bit_xor<>",
    }
)


def _canonical_operator_cpp(operator: CxxOperator) -> str:
    """Normalize supported type and construction spellings for recognition."""

    return operator.cpp.strip().replace("<T>", "<>").removesuffix("{}")


@dataclass(frozen=True, eq=False)
class GroupReduceSemantics:
    """Describe a CUB reduction and its scratch allocation policy."""

    primitive: ReduceSemantics
    cub_algorithm: BlockReduceAlgorithm | str | None = None
    storage_ownership: StorageOwnership = StorageOwnership.IMPLEMENTATION
    storage_sharing: str | None = None
    storage_size_in_bytes: int | None = None
    storage_alignment: int | None = None
    storage_auto_sync: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, ReduceSemantics):
            raise TypeError("primitive must be ReduceSemantics")
        if self.cub_algorithm is not None:
            try:
                algorithm = normalize_block_reduce_algorithm(self.cub_algorithm)
            except ValueError as exc:
                raise ValueError(
                    "unsupported CUB BlockReduce algorithm "
                    f"{self.cub_algorithm!r}"
                ) from exc
            object.__setattr__(self, "cub_algorithm", algorithm)
        object.__setattr__(
            self, "storage_ownership", StorageOwnership(self.storage_ownership)
        )
        if self.storage_ownership is StorageOwnership.NONE:
            raise ValueError("CUB reductions require scratch storage")
        if self.storage_sharing not in {None, "shared", "exclusive"}:
            raise ValueError("storage_sharing must be shared or exclusive")
        for name in ("storage_size_in_bytes", "storage_alignment"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if not isinstance(self.storage_auto_sync, bool):
            raise TypeError("storage_auto_sync must be a bool")
        if self.storage_ownership is StorageOwnership.IMPLEMENTATION:
            if any(
                value is not None
                for value in (
                    self.storage_sharing,
                    self.storage_size_in_bytes,
                    self.storage_alignment,
                )
            ):
                raise ValueError(
                    "implementation-owned storage cannot carry caller requests"
                )
        elif self.storage_sharing is None:
            raise ValueError("caller-owned storage requires storage_sharing")

    @property
    def dtype(self) -> Any:
        return self.primitive.dtype

    @property
    def operation(self) -> ReduceOperation:
        return self.primitive.operation

    @property
    def operand_kind(self) -> GroupOperandKind:
        return GroupOperandKind(self.primitive.value_kind.value)

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def valid_items(self) -> ArgumentBinding:
        return self.primitive.valid_items

    @property
    def reduce_operator(
        self,
    ) -> CxxOperator | PythonOperator | StatefulOperator | None:
        return self.primitive.reduce_operator

    @property
    def result_visibility(self) -> ResultVisibility:
        return ResultVisibility.GROUP_ROOT

    @property
    def returns_value(self) -> bool:
        return True

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.primitive.semantic_key,
            None if self.cub_algorithm is None else self.cub_algorithm.value,
            self.storage_ownership.value,
            self.storage_sharing,
            self.storage_size_in_bytes,
            self.storage_alignment,
            self.storage_auto_sync,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupReduceSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _call_classifications(
    operation: GroupReduceSemantics,
) -> tuple[ParameterClassification, ...]:
    """Separate input values from controls used to build the implementation.

    The payload is always a runtime input. Operator and count descriptions
    determine their own binding kinds. Algorithm selection remains static.
    """

    classifications = [
        ParameterClassification(
            "value", ArgumentKind.RUNTIME, ParameterRole.INPUT
        )
    ]
    operator = operation.reduce_operator
    if operator is not None:
        classifications.append(
            ParameterClassification(
                "binary_op", operator.argument_kind, operator.role
            )
        )
    if operation.valid_items.argument_kind is not None:
        classifications.append(
            ParameterClassification(
                "valid_items",
                operation.valid_items.argument_kind,
                (
                    ParameterRole.CONSTANT
                    if operation.valid_items.kind is BindingKind.STATIC
                    else ParameterRole.INPUT
                ),
            )
        )
    classifications.extend(
        (
            ParameterClassification(
                "algorithm", ArgumentKind.STATIC, ParameterRole.CONSTANT
            ),
        )
    )
    return tuple(classifications)


def _result_contract(operation: GroupReduceSemantics) -> ResultContract:
    """Define the reduced scalar only at group rank zero."""

    return ResultContract(
        (
            LogicalResultContract(
                name="value",
                dtype=operation.dtype,
                visibility=ResultVisibility.GROUP_ROOT,
                ownership=ResultOwnership.GROUP_ROOT,
                operand_kind=GroupOperandKind.SCALAR,
                items_per_member=1,
                root_rank=0,
            ),
        )
    )


def _has_proven_commutative_reduce_operator(
    operation: GroupReduceSemantics,
) -> bool:
    """Recognize operators safe for CUB's commutative-only algorithm.

    Use the known built-in set. A custom callback is not treated as
    commutative merely because it could implement the same operation.
    """

    if operation.operation is ReduceOperation.SUM:
        return True
    operator = operation.reduce_operator
    return isinstance(operator, CxxOperator) and (
        _canonical_operator_cpp(operator) in _COMMUTATIVE_REDUCE_OPERATORS
    )


def _plan_cub_reduce(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupReduceSemantics,
) -> GroupLoweringPlan:
    """Plan a direct CUB reduction and record its limits for the backend.

    CUB defines the result at the group root.
    Blocks and physical or logical warps support scalar or array inputs.
    Reject group shapes, algorithms, and operator combinations
    that this path cannot implement.

    A block request without an algorithm records ``WARP_REDUCTIONS`` in both
    the operation and the plan's call, making it part of the semantic key.

    Validate static prefix counts here. Runtime counts remain a caller
    precondition recorded in the core plan. The backend plans CUB scratch.
    Counts must be uniform across the group. Where a backend supports
    stateful callbacks, their runtime state must also be uniform; the core
    descriptor alone does not establish that support.
    """

    if resolved.kind not in {"block", "warp", "threads_within_warp"}:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "valid_items, explicit CUB algorithms, and custom operators are "
            "supported only for physical block, physical-warp, and "
            "logical-warp groups",
        )
    if (
        resolved.kind != "block"
        and operation.storage_ownership is StorageOwnership.CALLER
    ):
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "explicit temp_storage is supported only for block reductions; "
            "omit temp_storage for warp reductions",
        )

    if operation.valid_items.kind is BindingKind.STATIC:
        valid_items = operation.valid_items.value
        if isinstance(valid_items, bool) or not isinstance(
            valid_items, Integral
        ):
            raise TypeError("static valid_items must be an integer")
        valid_items = int(valid_items)
        group_size = resolved.static_size
        assert group_size is not None
        if valid_items < 1:
            raise ValueError("static valid_items must be at least 1")
        if valid_items > group_size:
            raise ValueError(
                f"static valid_items {valid_items} exceeds group size "
                f"{group_size}"
            )

    assert launch.exact_block_dim is not None
    reduce_operator = operation.reduce_operator
    if resolved.kind == "block":
        algorithm = (
            operation.cub_algorithm or BlockReduceAlgorithm.WARP_REDUCTIONS
        )
        if algorithm is BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "group BlockReduce does not expose "
                "BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC because its "
                "current CUB implementation is addition-specific",
            )
        if (
            algorithm is BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY
            and not _has_proven_commutative_reduce_operator(operation)
        ):
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY requires a reduction "
                "operator with proven commutativity",
            )
        if operation.cub_algorithm is None:
            operation = replace(operation, cub_algorithm=algorithm)
            call = GroupPrimitiveCall(group=call.group, operation=operation)
        specialization = make_block_reduce_specialization(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=operation.items_per_thread,
            operation=operation.operation,
            algorithm=algorithm,
            value_kind=ReduceValueKind(operation.operand_kind.value),
            reduce_operator=reduce_operator,
            valid_items=operation.valid_items,
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        cpp_class = "cub::BlockReduce"
        header = "cub/block/block_reduce.cuh"
    else:
        if operation.cub_algorithm is not None:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "CUB algorithm selection applies to BlockReduce, "
                "not WarpReduce",
            )
        warp_width = resolved.static_size
        assert warp_width is not None
        if warp_width < 17 and warp_width & (warp_width - 1):
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.GROUP_KIND,
                "CUB WarpReduce supports only one non-power-of-two group per "
                "physical warp; group_by widths below 17 would create "
                "multiple groups with an incompatible CUB synchronization mask",
            )
        operation_name = (
            WarpReduceOperation.SUM
            if operation.operation is ReduceOperation.SUM
            else WarpReduceOperation.REDUCE
        )
        specialization = make_warp_reduce_specialization(
            dtype=operation.dtype,
            threads_in_warp=warp_width,
            operation=operation_name,
            items_per_thread=operation.items_per_thread,
            value_kind=ReduceValueKind(operation.operand_kind.value),
            reduce_operator=reduce_operator,
            valid_items=operation.valid_items,
            include_full_warp=False,
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        cpp_class = "cub::WarpReduce"
        header = "cub/warp/warp_reduce.cuh"

    result = _result_contract(operation)
    requirements = _build_execution_requirements(
        resolved,
        launch,
        storage_ownership=operation.storage_ownership,
        cpp_type=None,
        storage_sharing=operation.storage_sharing,
        requested_size_in_bytes=operation.storage_size_in_bytes,
        requested_alignment=operation.storage_alignment,
        auto_sync=operation.storage_auto_sync,
        uniform_arguments=(
            *(
                ("binary_op",)
                if isinstance(reduce_operator, StatefulOperator)
                else ()
            ),
            *(("valid_items",) if operation.primitive.has_valid_items else ()),
        ),
        valid_member_selection=(
            "first N members by linear group rank"
            if operation.primitive.has_valid_items
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
            if operation.primitive.has_valid_items
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
    GroupReduceSemantics,
    classifications=_call_classifications,
    planner=_plan_cub_reduce,
    group_kinds=frozenset({"warp", "threads_within_warp", "block"}),
    unsupported_group_message=(
        "cuda.coop Reduce supports only block, physical-warp, and "
        "logical-warp groups"
    ),
)


__all__ = [
    "GroupReduceSemantics",
]
