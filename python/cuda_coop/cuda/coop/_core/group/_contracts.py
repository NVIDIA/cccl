# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cross-family participation, result, storage, and failure contracts.

Family planners call these helpers only after resolving a static thread group.
The helpers centralize cache-relevant contracts without owning any primitive's
semantic choices or a backend's compiler lifecycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._model import (
    ArgumentPrecondition,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupPrimitiveCall,
    LogicalResultContract,
    ParticipationContract,
    ResultContract,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    SynchronizationContract,
    SynchronizationScope,
    TempStorageContract,
    UnsupportedReason,
    UnsupportedReasonCode,
)

if TYPE_CHECKING:
    from ._dispatch import GroupOperationSemantics


def _unsupported(
    call: GroupPrimitiveCall,
    resolved_group: ThreadGroup,
    code: UnsupportedReasonCode,
    message: str,
) -> GroupLoweringPlan:
    return GroupLoweringPlan(
        target=GroupLoweringTarget.UNSUPPORTED,
        call=call,
        resolved_group=resolved_group,
        implementation=None,
        participation=None,
        result=None,
        synchronization=None,
        temp_storage=None,
        provenance=None,
        unsupported=UnsupportedReason(code=code, message=message),
    )


def _contracts(
    resolved_group: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupOperationSemantics,
    *,
    visibility: ResultVisibility,
    storage_ownership: StorageOwnership,
    cpp_type: str | None,
    storage_sharing: str | None = None,
    requested_size_in_bytes: int | None = None,
    requested_alignment: int | None = None,
    auto_sync: bool = True,
    uniform_arguments: tuple[str, ...] = (),
    valid_member_selection: str | None = None,
    argument_preconditions: tuple[ArgumentPrecondition, ...] = (),
    returns_value: bool = True,
) -> tuple[
    ParticipationContract,
    ResultContract | None,
    SynchronizationContract,
    TempStorageContract,
]:
    group_size = resolved_group.static_size
    assert group_size is not None
    if resolved_group.kind != "block":
        raise ValueError("Block Load/Store contracts require a block group")
    instances = 1
    index = "cta"
    barrier = SynchronizationScope.BLOCK if auto_sync else SynchronizationScope.NONE
    result_kind = GroupOperandKind.ARRAY
    result_items_per_member = operation.items_per_thread
    ownership = (
        ResultOwnership.GROUP_ROOT
        if visibility is ResultVisibility.GROUP_ROOT
        else ResultOwnership.EACH_MEMBER
    )
    results = []
    if returns_value:
        results.append(
            LogicalResultContract(
                name="value",
                dtype=operation.dtype,
                visibility=visibility,
                ownership=ownership,
                operand_kind=result_kind,
                items_per_member=result_items_per_member,
                root_rank=(0 if ownership is ResultOwnership.GROUP_ROOT else None),
            )
        )
    return (
        ParticipationContract(
            group_kind=resolved_group.kind,
            exact_group_size=group_size,
            exact_block_dim=launch.exact_block_dim,
            complete_membership=resolved_group.complete_membership is not False,
            contiguous=True,
            aligned=True,
            converged_entry=True,
            complete_parent_partition=(
                resolved_group.kind == "warp"
                or resolved_group.complete_membership is True
            ),
            uniform_arguments=uniform_arguments,
            valid_member_selection=valid_member_selection,
            argument_preconditions=argument_preconditions,
        ),
        ResultContract(tuple(results)) if results else None,
        SynchronizationContract(
            converged_entry=True,
            storage_reuse_barrier=barrier,
        ),
        TempStorageContract(
            ownership=storage_ownership,
            address_space="shared",
            cpp_type=cpp_type,
            instances=(
                None
                if storage_ownership is StorageOwnership.IMPLEMENTATION
                else instances
            ),
            instance_index=(
                None if storage_ownership is StorageOwnership.IMPLEMENTATION else index
            ),
            exact_layout_required=storage_ownership is StorageOwnership.CALLER,
            sharing=storage_sharing,
            requested_size_in_bytes=requested_size_in_bytes,
            requested_alignment=requested_alignment,
            auto_sync=auto_sync,
        ),
    )


__all__ = []
