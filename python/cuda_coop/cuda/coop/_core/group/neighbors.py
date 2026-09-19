# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block neighbor result, participation, and scratch contracts."""

from dataclasses import dataclass, field

from .._bindings import ArgumentBinding, BindingKind, _normalize_i32_binding
from .._types import ArgumentKind, ParameterClassification, ParameterRole
from ..block.neighbors import BlockNeighborSemantics, make_block_neighbor_spec
from ._contracts import _contracts
from ._dispatch import _register_group_operation_family
from ._model import (
    ArgumentPrecondition,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    ImplementationProvenance,
    LogicalResultContract,
    PreconditionEnforcement,
    ResultContract,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
)


@dataclass(frozen=True, eq=False)
class GroupNeighborSemantics:
    primitive: BlockNeighborSemantics
    valid_items: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)

    def __post_init__(self):
        if not isinstance(self.primitive, BlockNeighborSemantics):
            raise TypeError("primitive must be BlockNeighborSemantics")
        object.__setattr__(
            self,
            "valid_items",
            _normalize_i32_binding(self.valid_items, name="valid_items"),
        )
        if self.primitive.partial != (self.valid_items.kind is not BindingKind.OMITTED):
            raise ValueError("partial neighbor calls require a valid_items binding")

    @property
    def dtype(self):
        return self.primitive.dtype

    @property
    def result_visibility(self):
        return ResultVisibility.PER_MEMBER

    @property
    def returns_value(self):
        return True

    @property
    def semantic_key(self):
        return self.primitive.semantic_key, self.valid_items.semantic_key

    def __eq__(self, other):
        if not isinstance(other, GroupNeighborSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self):
        return hash(self.semantic_key)


def _classifications(operation):
    result = [
        ParameterClassification("values", ArgumentKind.RUNTIME, ParameterRole.INPUT)
    ]
    if operation.primitive.partial:
        result.append(
            ParameterClassification(
                "valid_items", operation.valid_items.argument_kind, ParameterRole.INPUT
            )
        )
    for flag, name in (
        (operation.primitive.predecessor, "tile_predecessor_item"),
        (operation.primitive.successor, "tile_successor_item"),
    ):
        if flag:
            result.append(
                ParameterClassification(name, ArgumentKind.RUNTIME, ParameterRole.INPUT)
            )
    result.append(
        ParameterClassification("operator", ArgumentKind.STATIC, ParameterRole.OPERATOR)
    )
    return tuple(result)


def _plan_neighbors(call, resolved, launch, operation):
    primitive = operation.primitive
    capacity = launch.exact_block_threads * primitive.items_per_thread
    if operation.valid_items.kind is BindingKind.STATIC:
        if not 0 <= operation.valid_items.value <= capacity:
            raise ValueError(
                f"valid_items must be between 0 and the block tile size ({capacity})"
            )
    result = ResultContract(
        tuple(
            LogicalResultContract(
                name=name,
                dtype=primitive.result_dtype,
                visibility=ResultVisibility.PER_MEMBER,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.ARRAY,
                items_per_member=primitive.items_per_thread,
            )
            for name in primitive.result_names
        )
    )
    uniform = []
    if primitive.partial:
        uniform.append("valid_items")
    if primitive.predecessor:
        uniform.append("tile_predecessor_item")
    if primitive.successor:
        uniform.append("tile_successor_item")
    contracts = _contracts(
        resolved,
        launch,
        result=result,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=tuple(uniform),
        argument_preconditions=(
            ArgumentPrecondition(
                name="valid_items",
                minimum=0,
                maximum=capacity,
                enforcement=PreconditionEnforcement.PLANNER_VALIDATED
                if operation.valid_items.kind is BindingKind.STATIC
                else PreconditionEnforcement.CALLER,
            ),
        )
        if primitive.partial
        else (),
    )
    spec = make_block_neighbor_spec(primitive, block_dim=launch.exact_block_dim)
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
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
            header=f"cub/block/block_{primitive.operation}.cuh",
            cpp_class="cub::BlockAdjacentDifference"
            if primitive.operation == "adjacent_difference"
            else "cub::BlockDiscontinuity",
            method=spec.metadata["method"],
        ),
    )


_register_group_operation_family(
    GroupNeighborSemantics,
    classifications=_classifications,
    planner=_plan_neighbors,
    group_kinds=frozenset({"block"}),
    unsupported_group_message="neighbor primitives require a complete block",
)

__all__ = ["GroupNeighborSemantics"]
