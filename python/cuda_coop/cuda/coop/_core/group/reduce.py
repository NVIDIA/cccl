# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable scalar BlockReduce and WarpReduce semantics and planning."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from ..warp.reduce import make_warp_reduce_spec
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
    """Backend-neutral semantics of one group-wide scalar reduction."""

    dtype: Any
    operation: GroupReduceOperation = GroupReduceOperation.REDUCE
    binary_op: GroupReduceOperator = GroupReduceOperator.SUM
    algorithm: GroupReduceAlgorithm | None = None
    valid_items: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", GroupReduceOperation(self.operation))
        object.__setattr__(
            self,
            "binary_op",
            normalize_block_reduce_operator(self.binary_op),
        )
        if self.algorithm is not None:
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
            None if self.algorithm is None else self.algorithm.value,
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
    """Plan one resolved scalar block or physical-warp reduction."""

    assert launch.exact_block_dim is not None
    assert launch.exact_block_threads is not None
    if resolved_group.kind == "block":
        group_size = launch.exact_block_threads
        implementation = make_block_reduce_spec(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            operation=operation.operation,
            binary_op=operation.binary_op,
            algorithm=operation.algorithm,
            valid_items=operation.has_valid_items,
        )
        if operation.algorithm is None:
            operation = replace(operation, algorithm=implementation.algorithm)
            call = GroupPrimitiveCall(
                group=call.group,
                operation=operation,
                source=call.source,
            )
        target = GroupLoweringTarget.CUB_BLOCK
        header = "cub/block/block_reduce.cuh"
        cpp_class = "cub::BlockReduce"
    elif resolved_group.kind == "warp":
        group_size = resolved_group.static_size
        assert group_size is not None
        if operation.algorithm is not None:
            raise ValueError(
                "cuda.coop reduction algorithm selection applies to block groups, "
                "not physical warps"
            )
        implementation = make_warp_reduce_spec(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            operation=operation.operation.value,
            binary_op=operation.binary_op,
            valid_items=operation.has_valid_items,
        )
        target = GroupLoweringTarget.CUB_WARP
        header = "cub/warp/warp_reduce.cuh"
        cpp_class = "cub::WarpReduce"
    else:
        raise ValueError(f"unsupported reduction group kind {resolved_group.kind!r}")
    _validate_static_valid_items(operation.valid_items, group_size=group_size)
    participation, result, synchronization, temp_storage = _reduction_contracts(
        resolved_group,
        launch,
        operation,
    )
    return GroupLoweringPlan(
        target=target,
        call=call,
        resolved_group=resolved_group,
        implementation=implementation,
        participation=participation,
        result=result,
        synchronization=synchronization,
        temp_storage=temp_storage,
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=cpp_class,
            method=implementation.method_name,
        ),
    )


__all__ = [
    "GroupReduceAlgorithm",
    "GroupReduceOperation",
    "GroupReduceOperator",
    "GroupReduceSemantics",
]
