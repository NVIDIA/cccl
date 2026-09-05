# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable scalar BlockReduce semantics and planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._bindings import ArgumentBinding, BindingKind, _normalize_i32_binding
from .._symbols import semantic_token
from ..block.reduce import (
    BlockReduceAlgorithm,
    BlockReduceOperation,
    BlockReduceOperator,
    make_block_reduce_spec,
    normalize_block_reduce_algorithm,
    normalize_block_reduce_operator,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._contracts import _reduction_contracts, _validate_static_valid_items
from ._model import (
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupPrimitiveCall,
    ImplementationProvenance,
)

GroupReduceAlgorithm = BlockReduceAlgorithm
GroupReduceOperation = BlockReduceOperation
GroupReduceOperator = BlockReduceOperator


@dataclass(frozen=True, eq=False)
class GroupReduceSemantics:
    """Backend-neutral semantics of one block-wide scalar reduction."""

    dtype: Any
    operation: GroupReduceOperation = GroupReduceOperation.REDUCE
    binary_op: GroupReduceOperator = GroupReduceOperator.SUM
    algorithm: GroupReduceAlgorithm = GroupReduceAlgorithm.WARP_REDUCTIONS
    valid_items: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", GroupReduceOperation(self.operation))
        object.__setattr__(
            self,
            "binary_op",
            normalize_block_reduce_operator(self.binary_op),
        )
        object.__setattr__(
            self,
            "algorithm",
            normalize_block_reduce_algorithm(self.algorithm),
        )
        if not isinstance(self.valid_items, ArgumentBinding):
            raise TypeError("valid_items must be an ArgumentBinding")
        object.__setattr__(
            self,
            "valid_items",
            _normalize_i32_binding(self.valid_items, name="valid_items"),
        )
        if self.operation is GroupReduceOperation.SUM and (
            self.binary_op is not GroupReduceOperator.SUM
        ):
            raise ValueError("cuda.coop.sum requires the sum operator")

    @property
    def has_valid_items(self) -> bool:
        return self.valid_items.kind is not BindingKind.OMITTED

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            f"group_{self.operation.value}",
            semantic_token(self.dtype),
            self.binary_op.value,
            self.algorithm.value,
            self.valid_items.semantic_key,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupReduceSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _plan_reduce(
    call: GroupPrimitiveCall,
    resolved_group: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupReduceSemantics,
) -> GroupLoweringPlan:
    """Plan one resolved scalar block reduction through CUB."""

    assert launch.exact_block_dim is not None
    assert launch.exact_block_threads is not None
    _validate_static_valid_items(
        operation.valid_items,
        group_size=launch.exact_block_threads,
    )
    implementation = make_block_reduce_spec(
        dtype=operation.dtype,
        block_dim=launch.exact_block_dim,
        operation=operation.operation,
        binary_op=operation.binary_op,
        algorithm=operation.algorithm,
        valid_items=operation.has_valid_items,
    )
    participation, result, synchronization, temp_storage = _reduction_contracts(
        resolved_group,
        launch,
        operation,
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved_group,
        implementation=implementation,
        participation=participation,
        result=result,
        synchronization=synchronization,
        temp_storage=temp_storage,
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_reduce.cuh",
            cpp_class="cub::BlockReduce",
            method=implementation.method_name,
        ),
    )


__all__ = [
    "GroupReduceAlgorithm",
    "GroupReduceOperation",
    "GroupReduceOperator",
    "GroupReduceSemantics",
]
