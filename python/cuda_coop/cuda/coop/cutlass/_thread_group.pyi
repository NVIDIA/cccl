# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type hints for cuda.coop.cutlass thread groups.

Overloads keep the group kind in the type, so a type checker can tell blocks,
physical warps, and logical warps apart. A warp partition returns a logical
thread group. A block partition returns warps within a block, which Load/Store
do not accept. Lowering checks group width and membership against the
enclosing kernel launch.
"""

from typing import Generic, Literal, TypeAlias, overload

from typing_extensions import TypeVar

from .. import ThreadHierarchy
from .._core.api.thread_group import ThreadGroup as CommonThreadGroup
from .._typing import ThreadGroupKind

_GroupKindT_co = TypeVar(
    "_GroupKindT_co",
    bound=ThreadGroupKind,
    covariant=True,
    default=ThreadGroupKind,
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
        """Describe groups with a compile-time count of threads per group.

        Load/Store accept widths 1, 2, 4, 8, 16, and 32. Each width divides a
        physical warp, so either exhaustive setting gives complete groups.
        """

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

__all__ = [
    "Hierarchy",
    "ThreadGroup",
    "ThreadHierarchy",
    "this_block",
    "this_warp",
]
