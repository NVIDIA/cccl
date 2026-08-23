# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TopK semantics and one-dimensional complete-block lowering.

TopK already has a fully specialized portable ``BlockTopKSpec`` because its
private-CUB compatibility shim is shared by both backends. This family wraps
that spec without rebuilding it so its semantic key remains the provider and
artifact identity. Frontend payload inference and runtime control validation
remain backend-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._bindings import BindingKind
from ..block.topk import BlockTopKSpec, TopKPayload
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._contracts import _contracts, _unsupported
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


@dataclass(frozen=True, eq=False)
class GroupTopKSemantics:
    """One existing BlockTopK specialization attached to a block group."""

    primitive: BlockTopKSpec

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockTopKSpec):
            raise TypeError("primitive must be a BlockTopKSpec")

    @property
    def dtype(self) -> Any:
        return self.primitive.specialization.template_arguments["KeyT"]

    @property
    def value_dtype(self) -> Any | None:
        if self.primitive.payload is TopKPayload.KEYS:
            return None
        return self.primitive.specialization.template_arguments["ValueT"]

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """Retain the BlockTopK semantic key used by existing providers."""

        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupTopKSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _topk_preconditions(
    operation: GroupTopKSemantics,
) -> tuple[ArgumentPrecondition, ...]:
    primitive = operation.primitive
    tile_size = primitive.tile_size
    preconditions = [
        ArgumentPrecondition(
            name="k",
            minimum=1,
            maximum=tile_size,
            enforcement=PreconditionEnforcement.CALLER,
        )
    ]
    if primitive.num_valid.kind is not BindingKind.OMITTED:
        preconditions.append(
            ArgumentPrecondition(
                name="num_valid",
                minimum=1,
                maximum=tile_size,
                enforcement=PreconditionEnforcement.CALLER,
            )
        )
    if primitive.begin_bit.kind is not BindingKind.OMITTED:
        preconditions.extend(
            (
                ArgumentPrecondition(
                    name="begin_bit",
                    minimum=0,
                    maximum=None,
                    enforcement=PreconditionEnforcement.CALLER,
                ),
                ArgumentPrecondition(
                    name="end_bit",
                    minimum=1,
                    maximum=None,
                    enforcement=PreconditionEnforcement.CALLER,
                ),
            )
        )
    return tuple(preconditions)


def _runtime_uniform_arguments(operation: GroupTopKSemantics) -> tuple[str, ...]:
    primitive = operation.primitive
    names = ["k"]
    for name, argument in (
        ("num_valid", primitive.num_valid),
        ("begin_bit", primitive.begin_bit),
        ("end_bit", primitive.end_bit),
    ):
        if argument.kind is BindingKind.RUNTIME:
            names.append(name)
    return tuple(names)


def _plan_topk(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupTopKSemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group TopK supports complete physical block groups",
        )
    assert launch.exact_block_dim is not None
    primitive = operation.primitive
    if launch.exact_block_dim != primitive.block_dim:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "BlockTopK specialization dimensions must match the exact launch "
            f"dimensions; received {primitive.block_dim!r} and "
            f"{launch.exact_block_dim!r}",
        )
    if primitive.block_dim[1:] != (1, 1):
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "group TopK supports one-dimensional blocks",
        )
    if primitive.block_dim[0] > 1024:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "group TopK block thread count must be <= 1024",
        )

    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=_runtime_uniform_arguments(operation),
        valid_member_selection="first k elements of the valid tile prefix",
        argument_preconditions=_topk_preconditions(operation),
        returns_value=False,
    )
    results = [
        LogicalResultContract(
            name="keys",
            dtype=operation.dtype,
            visibility=ResultVisibility.PER_MEMBER,
            ownership=ResultOwnership.EACH_MEMBER,
            operand_kind=GroupOperandKind.ARRAY,
            items_per_member=primitive.items_per_thread,
        )
    ]
    if primitive.payload is TopKPayload.PAIRS:
        results.append(
            LogicalResultContract(
                name="values",
                dtype=operation.value_dtype,
                visibility=ResultVisibility.PER_MEMBER,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.ARRAY,
                items_per_member=primitive.items_per_thread,
            )
        )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=primitive.specialization,
        participation=contracts[0],
        result=ResultContract(tuple(results)),
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_topk.cuh",
            cpp_class="cub::BlockTopKCoop",
            method=primitive.method_name,
        ),
    )


__all__ = ["GroupTopKSemantics"]
