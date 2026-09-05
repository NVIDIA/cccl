# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cross-family participation, result, storage, and failure contracts.

Family planners call these helpers only after resolving a static thread group.
The helpers centralize cache-relevant contracts without owning any primitive's
semantic choices or a backend's compiler lifecycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._types import StatefulOperator
from ..block.shuffle import BlockShuffleValueKind
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
    from .discontinuity import GroupDiscontinuitySemantics
    from .reduce import GroupReduceSemantics
    from .scan import GroupScanSemantics
    from .shuffle import GroupShuffleSemantics

    group_size = resolved_group.static_size
    assert group_size is not None
    if resolved_group.kind in {"warp", "threads_within_warp"}:
        logical_width = (
            32 if resolved_group.kind == "warp" else resolved_group.static_size
        )
        assert logical_width is not None
        instances = launch.exact_block_threads // logical_width  # type: ignore[operator]
        index = f"linear_thread_rank / {logical_width}"
        barrier = SynchronizationScope.WARP
    elif resolved_group.kind == "block":
        instances = 1
        index = "cta"
        barrier = SynchronizationScope.BLOCK
    elif resolved_group.kind == "thread":
        instances = 1
        index = "thread"
        barrier = SynchronizationScope.NONE
    else:
        instances = resolved_group.groups_per_parent or 1
        index = "group.rank(parent)"
        barrier = SynchronizationScope.GROUP
    if isinstance(operation, GroupReduceSemantics):
        result_kind = GroupOperandKind.SCALAR
        result_items_per_member = 1
    elif isinstance(operation, GroupScanSemantics):
        result_kind = operation.operand_kind
        result_items_per_member = operation.items_per_thread
    elif isinstance(operation, GroupShuffleSemantics):
        result_kind = (
            GroupOperandKind.ARRAY
            if operation.primitive.value_kind is BlockShuffleValueKind.ARRAY
            else GroupOperandKind.SCALAR
        )
        result_items_per_member = operation.items_per_thread
    else:
        result_kind = GroupOperandKind.ARRAY
        result_items_per_member = operation.items_per_thread
    ownership = (
        ResultOwnership.GROUP_ROOT
        if visibility is ResultVisibility.GROUP_ROOT
        else ResultOwnership.EACH_MEMBER
    )
    results = []
    if isinstance(operation, GroupDiscontinuitySemantics):
        for name, enabled in (
            ("head_flags", operation.primitive.has_heads),
            ("tail_flags", operation.primitive.has_tails),
        ):
            if enabled:
                results.append(
                    LogicalResultContract(
                        name=name,
                        dtype=operation.flag_dtype,
                        visibility=visibility,
                        ownership=ownership,
                        operand_kind=GroupOperandKind.ARRAY,
                        items_per_member=operation.items_per_thread,
                        root_rank=(
                            0 if ownership is ResultOwnership.GROUP_ROOT else None
                        ),
                    )
                )
    elif returns_value:
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
    if isinstance(operation, GroupShuffleSemantics):
        for name, enabled in (
            ("block_prefix", operation.primitive.block_prefix),
            ("block_suffix", operation.primitive.block_suffix),
        ):
            if enabled:
                results.append(
                    LogicalResultContract(
                        name=name,
                        dtype=operation.dtype,
                        visibility=ResultVisibility.ALL_MEMBERS,
                        ownership=ResultOwnership.EACH_MEMBER,
                        operand_kind=GroupOperandKind.SCALAR,
                        items_per_member=1,
                    )
                )
    if isinstance(operation, GroupScanSemantics) and operation.aggregate:
        results.append(
            LogicalResultContract(
                name="aggregate",
                dtype=operation.dtype,
                visibility=ResultVisibility.ALL_MEMBERS,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.SCALAR,
                items_per_member=1,
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
            address_space=(
                None
                if storage_ownership is StorageOwnership.IMPLEMENTATION
                else "shared"
            ),
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
        ),
    )


def _stateful_operator_uniformity(operator: Any) -> tuple[str, ...]:
    return ("operation",) if isinstance(operator, StatefulOperator) else ()


def _cub_warp_width(group: ThreadGroup) -> int:
    """Return a CUB-legal physical or logical warp width.

    The physical-warp descriptor always lowers at the architectural width.
    Mapped thread groups are more permissive than CUB, so reject widths that
    CUB cannot instantiate before constructing a low-level specialization.
    """

    if group.kind == "warp":
        return 32
    if group.kind != "threads_within_warp":
        raise ValueError("CUB warp primitives require a warp-based group")
    width = group.static_size
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width < 1
        or width > 32
        or width & (width - 1)
        or 32 % width != 0
    ):
        raise ValueError(
            "CUB-backed logical-warp operations require a power-of-two group "
            "width in [1, 32] that divides the 32-thread physical warp; "
            f"got {width!r}"
        )
    return width


def _unsupported_cub_warp_width(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
) -> tuple[int | None, GroupLoweringPlan | None]:
    try:
        return _cub_warp_width(resolved), None
    except ValueError as exc:
        return None, _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            str(exc),
        )


__all__ = []
