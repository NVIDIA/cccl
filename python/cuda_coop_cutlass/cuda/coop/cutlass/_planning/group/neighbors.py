# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Attach block participation and result contracts to neighbor operations.

The primitive record selects a CUB overload. The group record adds the
binding for ``valid_items``, and the planner checks it against the exact tile
size. The resulting plan tells frontends who owns each output and which count
or outside-tile neighbor arguments must agree across the block.
"""

from dataclasses import dataclass, field

from .._bindings import ArgumentBinding, BindingKind, normalize_i32_binding
from .._types import ArgumentKind, ParameterClassification, ParameterRole
from ..block.neighbors import (
    BlockNeighborSemantics,
    make_block_neighbor_specialization,
)
from ._dispatch import _register_group_operation_family
from ._execution_requirements import _build_execution_requirements
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
    """Pair a neighbor primitive with its omitted, static or runtime count.

    The count binding must be present exactly when the primitive selects a
    partial tile. Known counts receive integer validation here and tile-size
    validation in the planner, where the block dimensions are available. Cache
    identity includes the binding kind and any static value, rather than a
    compiler expression that supplies a runtime count.
    """

    primitive: BlockNeighborSemantics
    valid_items: ArgumentBinding = field(
        default_factory=ArgumentBinding.omitted
    )

    def __post_init__(self):
        """Match the count binding to the partial-tile overload."""

        if not isinstance(self.primitive, BlockNeighborSemantics):
            raise TypeError("primitive must be BlockNeighborSemantics")
        object.__setattr__(
            self,
            "valid_items",
            normalize_i32_binding(self.valid_items, name="valid_items"),
        )
        if self.primitive.partial != (
            self.valid_items.kind is not BindingKind.OMITTED
        ):
            raise ValueError(
                "partial neighbor calls require a valid_items binding"
            )

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
    """Label payloads and optional controls for the group call contract.

    Values and supplied neighbors are runtime inputs. A partial count keeps
    its static or runtime binding kind, while the binary operator is fixed
    during specialization. Omitted neighbor controls do not become public
    operands.
    """

    result = [
        ParameterClassification(
            "values", ArgumentKind.RUNTIME, ParameterRole.INPUT
        )
    ]
    if operation.primitive.partial:
        result.append(
            ParameterClassification(
                "valid_items",
                operation.valid_items.argument_kind,
                ParameterRole.INPUT,
            )
        )
    for flag, name in (
        (operation.primitive.predecessor, "tile_predecessor_item"),
        (operation.primitive.successor, "tile_successor_item"),
    ):
        if flag:
            result.append(
                ParameterClassification(
                    name, ArgumentKind.RUNTIME, ParameterRole.INPUT
                )
            )
    result.append(
        ParameterClassification(
            "operator", ArgumentKind.STATIC, ParameterRole.OPERATOR
        )
    )
    return tuple(result)


def _plan_neighbors(call, resolved, launch, operation):
    """Build result and participation contracts for the exact block tile.

    Check a known valid count against tile capacity, and record that bound as
    a precondition for runtime counts. The C++ adapter also checks runtime
    counts before narrowing them. Every member owns its portion of each
    blocked output; differences keep the input dtype and head/tail flags use
    int32.

    Count and supplied neighbor values must be block-uniform. Shared contracts
    supply implementation-owned scratch and its synchronization requirements;
    the frontend can replace those storage choices for an explicit descriptor.
    """

    primitive = operation.primitive
    capacity = launch.exact_block_threads * primitive.items_per_thread
    if (
        operation.valid_items.kind is BindingKind.STATIC
        and not 0 <= operation.valid_items.value <= capacity
    ):
        raise ValueError(
            "valid_items must be between 0 and the block tile size "
            f"({capacity})"
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
    requirements = _build_execution_requirements(
        resolved,
        launch,
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
    specialization = make_block_neighbor_specialization(
        primitive, block_dim=launch.exact_block_dim
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
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
            header=f"cub/block/block_{primitive.operation}.cuh",
            cpp_class="cub::BlockAdjacentDifference"
            if primitive.operation == "adjacent_difference"
            else "cub::BlockDiscontinuity",
            method=specialization.metadata["method"],
        ),
    )


_register_group_operation_family(
    GroupNeighborSemantics,
    classifications=_classifications,
    planner=_plan_neighbors,
    group_kinds=frozenset({"block"}),
    unsupported_group_message="neighbor primitives require a complete block",
)

__all__ = [
    "GroupNeighborSemantics",
]
