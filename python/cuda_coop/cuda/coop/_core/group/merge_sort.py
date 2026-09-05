# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Merge-sort semantics and block or warp lowering.

This family module owns the normalized keys/pairs result shape and chooses the
matching CUB specialization for a resolved group. It does not manage backend
activation, compiler state, or provider caches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._types import RuntimeValue
from ..block.merge_sort import BlockMergeSortSemantics, make_block_merge_sort_spec
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ..warp.merge_sort import make_warp_merge_sort_spec
from ._contracts import _contracts, _unsupported, _unsupported_cub_warp_width
from ._model import (
    ArgumentPrecondition,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupPrimitiveCall,
    ImplementationProvenance,
    PreconditionEnforcement,
    ResultVisibility,
    StorageOwnership,
    UnsupportedReasonCode,
)


@dataclass(frozen=True, eq=False)
class GroupMergeSortSemantics:
    """CUB MergeSort semantics attached to an explicit thread group."""

    primitive: BlockMergeSortSemantics

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockMergeSortSemantics):
            raise TypeError("primitive must be BlockMergeSortSemantics")

    @property
    def dtype(self) -> Any:
        return self.primitive.key_dtype

    @property
    def key_dtype(self) -> Any:
        return self.primitive.key_dtype

    @property
    def value_dtype(self) -> Any | None:
        return self.primitive.value_dtype

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupMergeSortSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _plan_merge_sort(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupMergeSortSemantics,
) -> GroupLoweringPlan:
    if resolved.kind not in {"block", "warp", "threads_within_warp"}:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group merge_sort supports complete block, physical-warp, and "
            "logical-warp groups",
        )

    primitive = operation.primitive
    assert launch.exact_block_dim is not None
    block_threads = launch.exact_block_threads
    assert block_threads is not None
    if resolved.kind == "block":
        if block_threads & (block_threads - 1):
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "cub::BlockMergeSort requires a power-of-two block thread count",
            )
        runtime_valid_items = (
            RuntimeValue("valid_items") if primitive.has_partial_tile else None
        )
        runtime_oob_default = (
            RuntimeValue("oob_default") if primitive.has_partial_tile else None
        )
        spec = make_block_merge_sort_spec(
            key_dtype=primitive.key_dtype,
            value_dtype=primitive.value_dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=primitive.items_per_thread,
            compare_operator=primitive.compare_operator,
            valid_items=runtime_valid_items,
            oob_default=runtime_oob_default,
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        cpp_class = "cub::BlockMergeSort"
        header = "cub/block/block_merge_sort.cuh"
        tile_threads = block_threads
    else:
        logical_width, width_error = _unsupported_cub_warp_width(call, resolved)
        if width_error is not None:
            return width_error
        assert logical_width is not None
        if block_threads % logical_width:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP,
                "warp merge_sort requires the exact block thread count to be "
                "a multiple of the logical warp width",
            )
        spec = make_warp_merge_sort_spec(
            key_dtype=primitive.key_dtype,
            value_dtype=primitive.value_dtype,
            items_per_thread=primitive.items_per_thread,
            threads_in_warp=logical_width,
            compare_operator=primitive.compare_operator,
            valid_items=(
                RuntimeValue("valid_items") if primitive.has_partial_tile else None
            ),
            oob_default=(
                RuntimeValue("oob_default") if primitive.has_partial_tile else None
            ),
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        cpp_class = "cub::WarpMergeSort"
        header = "cub/warp/warp_merge_sort.cuh"
        tile_threads = logical_width

    argument_preconditions = ()
    uniform_arguments = ()
    valid_member_selection = None
    if primitive.has_partial_tile:
        uniform_arguments = ("valid_items", "oob_default")
        valid_member_selection = "first valid_items tile elements"
        argument_preconditions = (
            ArgumentPrecondition(
                name="valid_items",
                minimum=0,
                maximum=tile_threads * primitive.items_per_thread,
                enforcement=PreconditionEnforcement.CALLER,
            ),
        )
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=uniform_arguments,
        valid_member_selection=valid_member_selection,
        argument_preconditions=argument_preconditions,
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


__all__ = ["GroupMergeSortSemantics"]
