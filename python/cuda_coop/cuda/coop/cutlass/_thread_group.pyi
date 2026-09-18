# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for CUTLASS group descriptors."""

from typing import Generic, Literal, TypeAlias, overload

from typing_extensions import TypeVar

from .. import ThreadHierarchy
from .._core.api.thread_group import ThreadGroup as CommonThreadGroup
from .._typing import ThreadGroupKind

_GroupKindT_co = TypeVar(
    "_GroupKindT_co", bound=ThreadGroupKind, covariant=True, default=ThreadGroupKind
)

Hierarchy = ThreadHierarchy

class ThreadGroup(CommonThreadGroup[_GroupKindT_co], Generic[_GroupKindT_co]):
    """A common group descriptor consumed by CUTLASS primitives."""

    @overload
    def group_by(
        self: ThreadGroup[Literal["warp"]],
        count: int,
        *,
        exhaustive: bool = True,
    ) -> ThreadGroup[Literal["threads_within_warp"]]:
        """Partition a physical warp into groups of threads."""

    @overload
    def group_by(
        self: ThreadGroup[Literal["block"]],
        count: int,
        *,
        exhaustive: bool = True,
    ) -> ThreadGroup[Literal["warps_within_block"]]:
        """Partition a block into groups of physical warps."""

BlockGroup: TypeAlias = ThreadGroup[Literal["block"]]
WarpGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp"]]
MemoryGroup: TypeAlias = BlockGroup | WarpGroup

def this_block() -> BlockGroup:
    """Describe the current CUDA thread block."""

def this_warp() -> ThreadGroup[Literal["warp"]]:
    """Describe the current complete physical warp."""

__all__ = ["Hierarchy", "ThreadGroup", "ThreadHierarchy", "this_block", "this_warp"]
