# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cross-family participation, result, storage, and failure contracts.

Family planners call these helpers only after resolving a static thread group.
The helpers centralize cache-relevant contracts without owning primitive
semantics or a backend compiler lifecycle.
"""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

from .._bindings import ArgumentBinding, BindingKind
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._model import (
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupPrimitiveCall,
    ParticipationContract,
    ResultContract,
    StorageOwnership,
    SynchronizationContract,
    SynchronizationScope,
    TempStorageContract,
    ThreadGroupResolution,
)

if TYPE_CHECKING:
    from .reduce import GroupReduceSemantics


def _unsupported_plan(
    call: GroupPrimitiveCall,
    resolution: ThreadGroupResolution,
) -> GroupLoweringPlan:
    assert resolution.unsupported is not None
    return GroupLoweringPlan(
        target=GroupLoweringTarget.UNSUPPORTED,
        call=call,
        resolved_group=resolution.group,
        implementation=None,
        participation=None,
        result=None,
        synchronization=None,
        temp_storage=None,
        provenance=None,
        unsupported=resolution.unsupported,
    )


def _validate_static_valid_items(
    binding: ArgumentBinding,
    *,
    group_size: int | None = None,
) -> None:
    if binding.kind is not BindingKind.STATIC:
        return
    value = binding.value
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"valid_items must be an integer, not {type(value).__name__}")
    normalized = int(value)
    if normalized < 1:
        raise ValueError("valid_items must be at least 1")
    if group_size is not None and normalized > group_size:
        raise ValueError(f"valid_items must be at most {group_size}")


def _reduction_contracts(
    resolved_group: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupReduceSemantics,
) -> tuple[
    ParticipationContract,
    ResultContract,
    SynchronizationContract,
    TempStorageContract,
]:
    """Build the portable contracts shared by block reduction providers."""

    assert launch.exact_block_dim is not None
    assert launch.exact_block_threads is not None
    return (
        ParticipationContract(
            group_kind=resolved_group.kind,
            exact_group_size=launch.exact_block_threads,
            exact_block_dim=launch.exact_block_dim,
            complete_membership=True,
            converged_entry=True,
            uniform_arguments=("valid_items",) if operation.has_valid_items else (),
            valid_member_selection=(
                "first valid_items block members" if operation.has_valid_items else None
            ),
        ),
        ResultContract(dtype=operation.dtype),
        SynchronizationContract(
            converged_entry=True,
            storage_reuse_barrier=SynchronizationScope.BLOCK,
        ),
        TempStorageContract(ownership=StorageOwnership.IMPLEMENTATION),
    )


__all__: list[str] = []
