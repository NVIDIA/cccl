# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve portable thread groups against compiler-verified launch facts."""

from __future__ import annotations

from ..launch import LaunchFacts
from ..thread_group import ThreadGroup, ThreadHierarchy
from ._model import (
    ThreadGroupResolution,
    UnsupportedReason,
    UnsupportedReasonCode,
)


def resolve_thread_group(
    group: ThreadGroup,
    launch: LaunchFacts,
) -> ThreadGroupResolution:
    """Resolve the current block against exact compiler launch facts."""

    if not isinstance(group, ThreadGroup):
        raise TypeError("group must be a ThreadGroup")
    if not isinstance(launch, LaunchFacts):
        raise TypeError("launch must be LaunchFacts")
    if launch.exact_block_dim is None:
        return ThreadGroupResolution(
            group,
            UnsupportedReason(
                UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM,
                "block reduction requires exact block dimensions",
            ),
        )
    if not launch.is_verified("exact_block_dim"):
        return ThreadGroupResolution(
            group,
            UnsupportedReason(
                UnsupportedReasonCode.UNVERIFIED_EXACT_BLOCK_DIM,
                "block reduction requires compiler-verified exact block dimensions",
            ),
        )
    existing = group.hierarchy.block_dim
    if existing is not None and existing != launch.exact_block_dim:
        raise ValueError(
            f"group block dimensions {existing!r} do not match exact launch "
            f"dimensions {launch.exact_block_dim!r}"
        )
    resolved = group.with_hierarchy(
        ThreadHierarchy(block_dim=launch.exact_block_dim),
        source="launch_facts",
    )
    return ThreadGroupResolution(resolved)


__all__ = ["resolve_thread_group"]
