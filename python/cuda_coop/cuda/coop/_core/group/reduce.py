# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reduction semantics and lowering for explicit thread groups.

The planner selects CUDAX for portable scalar reductions and CUB for qualified
array or algorithm-specific forms. Backend compiler and renderer lifecycle
remain outside this module.
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
    make_block_reduce_spec,
    normalize_block_reduce_algorithm,
)
from ..launch import LaunchFacts
from ..reduce import ReduceOperation, ReduceSemantics, ReduceValueKind
from ..thread_group import ThreadGroup
from ..warp.reduce import WarpReduceOperation, make_warp_reduce_spec
from ._contracts import (
    _contracts,
    _stateful_operator_uniformity,
    _unsupported,
    _unsupported_cub_warp_width,
)
from ._model import (
    ArgumentPrecondition,
    CudaxCallDescription,
    CudaxReturnKind,
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


@dataclass(frozen=True, eq=False)
class GroupReduceSemantics:
    primitive: ReduceSemantics
    broadcast: bool = True
    cub_algorithm: BlockReduceAlgorithm | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, ReduceSemantics):
            raise TypeError("primitive must be ReduceSemantics")
        if not isinstance(self.broadcast, bool):
            raise TypeError("broadcast must be a bool")
        if self.cub_algorithm is not None:
            try:
                algorithm = normalize_block_reduce_algorithm(self.cub_algorithm)
            except ValueError as exc:
                raise ValueError(
                    f"unsupported CUB BlockReduce algorithm {self.cub_algorithm!r}"
                ) from exc
            object.__setattr__(self, "cub_algorithm", algorithm)

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
    def reduce_operator(self) -> CxxOperator | PythonOperator | StatefulOperator | None:
        return self.primitive.reduce_operator

    @property
    def requests_cub(self) -> bool:
        return self.cub_algorithm is not None or self.primitive.has_valid_items

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.primitive.semantic_key,
            self.broadcast,
            None if self.cub_algorithm is None else self.cub_algorithm.value,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupReduceSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


_KNOWN_COMMUTATIVE_REDUCE_OPERATORS = frozenset(
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


def _has_proven_commutative_reduce_operator(
    operation: GroupReduceSemantics,
) -> bool:
    if operation.operation is ReduceOperation.SUM:
        return True
    operator = operation.reduce_operator
    if not isinstance(operator, CxxOperator):
        return False
    cpp = operator.cpp.replace("<T>", "<>").removesuffix("{}")
    return cpp in _KNOWN_COMMUTATIVE_REDUCE_OPERATORS


def _plan_reduce(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupReduceSemantics,
) -> GroupLoweringPlan:
    if operation.requests_cub and resolved.kind not in {
        "block",
        "warp",
        "threads_within_warp",
    }:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "valid_items and explicit CUB algorithms are supported only for "
            "physical block, physical-warp, and logical-warp groups",
        )
    if operation.requests_cub and operation.broadcast:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.CUB_BROADCAST,
            "direct CUB reduce returns a defined value only at the group root; "
            "it cannot satisfy broadcast=True",
        )
    if (
        resolved.kind == "block"
        and operation.requests_cub
        and operation.cub_algorithm is None
    ):
        operation = replace(
            operation,
            cub_algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
        )

    if not operation.requests_cub:
        implementation = CudaxCallDescription(
            primitive="reduce",
            header="cuda/experimental/coop.cuh",
            namespace="cuda::experimental::coop",
            overload="broadcasted" if operation.broadcast else "root_only",
            parameters=(
                *(
                    ParameterClassification(
                        f"item{index}",
                        ArgumentKind.RUNTIME,
                        ParameterRole.INPUT,
                    )
                    for index in range(operation.items_per_thread)
                ),
                *(
                    classification
                    for classification in call.argument_classifications
                    if classification.kind is ArgumentKind.RUNTIME
                    and classification.name != "value"
                ),
            ),
            return_kind=(
                CudaxReturnKind.VALUE
                if operation.broadcast
                else CudaxReturnKind.OPTIONAL_VALUE
            ),
        )
        contracts = _contracts(
            resolved,
            launch,
            operation,
            visibility=(
                ResultVisibility.ALL_MEMBERS
                if operation.broadcast
                else ResultVisibility.GROUP_ROOT
            ),
            storage_ownership=StorageOwnership.IMPLEMENTATION,
            cpp_type=None,
            uniform_arguments=_stateful_operator_uniformity(operation.reduce_operator),
        )
        return GroupLoweringPlan(
            target=GroupLoweringTarget.CUDAX_GROUP,
            call=call,
            resolved_group=resolved,
            implementation=implementation,
            participation=contracts[0],
            result=contracts[1],
            synchronization=contracts[2],
            temp_storage=contracts[3],
            provenance=ImplementationProvenance(
                library="CUDAX",
                header=implementation.header,
                cpp_class=implementation.namespace,
                method="reduce",
            ),
        )

    assert launch.exact_block_dim is not None
    operation_name: Any
    if operation.operation is ReduceOperation.SUM:
        operation_name = (
            ReduceOperation.SUM if resolved.kind == "block" else WarpReduceOperation.SUM
        )
    else:
        operation_name = (
            ReduceOperation.REDUCE
            if resolved.kind == "block"
            else WarpReduceOperation.REDUCE
        )
    reduce_operator = operation.reduce_operator

    if operation.valid_items.kind is BindingKind.STATIC:
        valid_items = operation.valid_items.value
        if isinstance(valid_items, bool) or not isinstance(valid_items, Integral):
            raise TypeError("static valid_items must be an integer")
        valid_items = int(valid_items)
        group_size = resolved.static_size
        assert group_size is not None
        if valid_items < 1:
            raise ValueError("static valid_items must be at least 1")
        if valid_items > group_size:
            raise ValueError(
                f"static valid_items {valid_items} exceeds group size {group_size}"
            )
    if resolved.kind == "block":
        algorithm = operation.cub_algorithm or BlockReduceAlgorithm.WARP_REDUCTIONS
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
        spec = make_block_reduce_spec(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=operation.items_per_thread,
            operation=operation_name,
            algorithm=algorithm,
            value_kind=ReduceValueKind(operation.operand_kind.value),
            reduce_operator=reduce_operator,
            valid_items=operation.valid_items,
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        cpp_class = "cub::BlockReduce"
        header = "cub/block/block_reduce.cuh"
    else:
        warp_width, width_error = _unsupported_cub_warp_width(call, resolved)
        if width_error is not None:
            return width_error
        assert warp_width is not None
        if operation.operand_kind is GroupOperandKind.ARRAY:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERAND_FORM,
                "direct CUB WarpReduce planning currently supports scalar operands",
            )
        if operation.cub_algorithm is not None:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "CUB algorithm selection applies to BlockReduce, not WarpReduce",
            )
        spec = make_warp_reduce_spec(
            dtype=operation.dtype,
            threads_in_warp=warp_width,
            operation=operation_name,
            reduce_operator=reduce_operator,
            valid_items=operation.valid_items,
            include_full_warp=False,
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        cpp_class = "cub::WarpReduce"
        header = "cub/warp/warp_reduce.cuh"
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.GROUP_ROOT,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=(
            *_stateful_operator_uniformity(operation.reduce_operator),
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


__all__ = ["GroupReduceSemantics"]
