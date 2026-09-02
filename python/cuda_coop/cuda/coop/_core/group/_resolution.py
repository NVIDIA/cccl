# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve portable thread groups against compiler-verified launch facts."""

from __future__ import annotations

from ..launch import LaunchFacts
from ..thread_group import PHYSICAL_WARP_THREADS, ThreadGroup, ThreadHierarchy
from ._model import (
    ThreadGroupResolution,
    UnsupportedReason,
    UnsupportedReasonCode,
)


def resolve_thread_group(
    group: ThreadGroup,
    launch: LaunchFacts,
) -> ThreadGroupResolution:
    """Resolve a current block or warp against exact compiler launch facts."""

    if not isinstance(group, ThreadGroup):
        raise TypeError("group must be a ThreadGroup")
    if not isinstance(launch, LaunchFacts):
        raise TypeError("launch must be LaunchFacts")
    if launch.exact_block_dim is None:
        return ThreadGroupResolution(
            group,
            UnsupportedReason(
                UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM,
                "cuda.coop reduction requires exact block dimensions",
            ),
        )
    if not launch.is_verified("exact_block_dim"):
        return ThreadGroupResolution(
            group,
            UnsupportedReason(
                UnsupportedReasonCode.UNVERIFIED_EXACT_BLOCK_DIM,
                "cuda.coop reduction requires compiler-verified exact block dimensions",
            ),
        )
    existing = group.hierarchy.block_dim
    if existing is not None and existing != launch.exact_block_dim:
        raise ValueError(
            f"group block dimensions {existing!r} do not match exact launch "
            f"dimensions {launch.exact_block_dim!r}"
        )
    if group.kind == "warp" and launch.exact_block_threads % PHYSICAL_WARP_THREADS != 0:
        return ThreadGroupResolution(
            group,
            UnsupportedReason(
                UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP,
                "physical-warp reduction requires an enclosing block composed "
                "of complete 32-thread warps; got "
                f"{launch.exact_block_threads} block threads",
            ),
        )
    resolved = group.with_hierarchy(
        ThreadHierarchy(block_dim=launch.exact_block_dim),
        source="launch_facts",
    )
    return ThreadGroupResolution(resolved)


__all__ = ["resolve_thread_group"]
