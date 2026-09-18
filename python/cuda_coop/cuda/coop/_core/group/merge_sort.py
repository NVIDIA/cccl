# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Out-of-place Merge Sort semantics for complete block and warp groups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._bindings import ArgumentBinding, BindingKind, _normalize_i32_binding
from .._types import ArgumentKind, ParameterClassification, ParameterRole
from ..block.merge_sort import BlockMergeSortSemantics, make_block_merge_sort_spec
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ..warp.merge_sort import make_warp_merge_sort_spec
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


@dataclass(frozen=True, eq=False)
class GroupMergeSortSemantics:
    primitive: BlockMergeSortSemantics
    valid_items: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockMergeSortSemantics):
            raise TypeError("primitive must be BlockMergeSortSemantics")
        if not isinstance(self.valid_items, ArgumentBinding):
            raise TypeError("valid_items must be an ArgumentBinding")
        object.__setattr__(
            self,
            "valid_items",
            _normalize_i32_binding(self.valid_items, name="valid_items"),
        )
        if self.primitive.has_partial_tile != (
            self.valid_items.kind is not BindingKind.OMITTED
        ):
            raise ValueError("partial Merge Sort requires a valid_items binding")

    @property
    def dtype(self) -> Any:
        return self.primitive.key_dtype

    @property
    def result_visibility(self) -> ResultVisibility:
        return ResultVisibility.PER_MEMBER

    @property
    def returns_value(self) -> bool:
        return True

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key, self.valid_items.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupMergeSortSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _classifications(operation):
    result = [
        ParameterClassification("keys", ArgumentKind.RUNTIME, ParameterRole.INPUT)
    ]
    if operation.primitive.has_values:
        result.append(
            ParameterClassification("values", ArgumentKind.RUNTIME, ParameterRole.INPUT)
        )
    result.append(
        ParameterClassification(
            "compare_op", ArgumentKind.STATIC, ParameterRole.OPERATOR
        )
    )
    if operation.primitive.has_partial_tile:
        result.extend(
            (
                ParameterClassification(
                    "valid_items",
                    operation.valid_items.argument_kind,
                    ParameterRole.INPUT,
                ),
                ParameterClassification(
                    "oob_default", ArgumentKind.RUNTIME, ParameterRole.INPUT
                ),
            )
        )
    return tuple(result)


def _plan_merge_sort(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupMergeSortSemantics,
) -> GroupLoweringPlan:
    primitive = operation.primitive
    kwargs = dict(
        key_dtype=primitive.key_dtype,
        value_dtype=primitive.value_dtype,
        items_per_thread=primitive.items_per_thread,
        compare_operator=primitive.compare_operator,
        valid_items=0 if primitive.has_partial_tile else None,
        oob_default=0 if primitive.has_partial_tile else None,
    )
    assert launch.exact_block_dim is not None
    if resolved.kind == "block":
        width = launch.exact_block_threads
        assert width is not None
        if width & (width - 1):
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "cub::BlockMergeSort requires a power-of-two block thread count",
            )
        spec = make_block_merge_sort_spec(
            block_dim=launch.exact_block_dim, **kwargs
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        header = "cub/block/block_merge_sort.cuh"
        cpp_class = "cub::BlockMergeSort"
    else:
        width, error = _unsupported_cub_warp_width(call, resolved)
        if error is not None:
            return error
        assert width is not None
        spec = make_warp_merge_sort_spec(threads_in_warp=width, **kwargs).specialization
        target = GroupLoweringTarget.CUB_WARP
        header = "cub/warp/warp_merge_sort.cuh"
        cpp_class = "cub::WarpMergeSort"
    capacity = width * primitive.items_per_thread
    if operation.valid_items.kind is BindingKind.STATIC:
        if not 0 <= operation.valid_items.value <= capacity:
            raise ValueError(
                f"valid_items must be between 0 and the group tile size ({capacity})"
            )
    outputs = [("keys", primitive.key_dtype)]
    if primitive.has_values:
        outputs.append(("values", primitive.value_dtype))
    result = ResultContract(
        tuple(
            LogicalResultContract(
                name=name,
                dtype=dtype,
                visibility=ResultVisibility.PER_MEMBER,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.ARRAY,
                items_per_member=primitive.items_per_thread,
            )
            for name, dtype in outputs
        )
    )
    contracts = _contracts(
        resolved,
        launch,
        result=result,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=("valid_items", "oob_default")
        if primitive.has_partial_tile
        else (),
        argument_preconditions=(
            ArgumentPrecondition(
                name="valid_items",
                minimum=0,
                maximum=capacity,
                enforcement=(
                    PreconditionEnforcement.PLANNER_VALIDATED
                    if operation.valid_items.kind is BindingKind.STATIC
                    else PreconditionEnforcement.CALLER
                ),
            ),
        )
        if primitive.has_partial_tile
        else (),
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
            library="CUB", header=header, cpp_class=cpp_class, method="Sort"
        ),
    )


_register_group_operation_family(
    GroupMergeSortSemantics,
    classifications=_classifications,
    planner=_plan_merge_sort,
    group_kinds=frozenset({"block", "warp", "threads_within_warp"}),
    unsupported_group_message="Merge Sort supports complete block, physical warp, and power-of-two logical warp groups",
)

__all__ = ["GroupMergeSortSemantics"]
