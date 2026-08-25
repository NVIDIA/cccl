# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for CUTLASS thread-hierarchy descriptors."""

from typing import Any, Generic, Literal, TypeAlias, overload

from typing_extensions import TypeVar

from .. import ThreadGroup as _CommonThreadGroup
from .. import ThreadHierarchy as ThreadHierarchy
from .._typing import CompilerIntegerLike, ThreadGroupKind, ThreadLevel
from ._typing import ScalarT

_GroupKindT_co = TypeVar(
    "_GroupKindT_co",
    bound=ThreadGroupKind,
    covariant=True,
    default=ThreadGroupKind,
)

Hierarchy = ThreadHierarchy

class ThreadGroup(
    _CommonThreadGroup[_GroupKindT_co],
    Generic[_GroupKindT_co],
):
    """Common CUDA group descriptor with CUTLASS lowering methods."""

    def rank(self, level: ThreadLevel = "thread") -> CompilerIntegerLike:
        """Return this group's rank as a CUTLASS ``Int32`` scalar."""

    def count(self, level: ThreadLevel = "thread") -> CompilerIntegerLike:
        """Return this group's count as a CUTLASS ``Int32`` scalar."""

    @overload
    def rank_as(self, dtype: type[ScalarT], level: ThreadLevel = "thread") -> ScalarT:
        """Convert rank to a portable or structural CUTLASS numeric dtype."""

    @overload
    def rank_as(self, dtype: None = None, level: ThreadLevel = "thread") -> Any:
        """Omit dtype or use an ``Any``-typed external CUTLASS dtype token."""

    @overload
    def count_as(self, dtype: type[ScalarT], level: ThreadLevel = "thread") -> ScalarT:
        """Convert count to a portable or structural CUTLASS numeric dtype."""

    @overload
    def count_as(self, dtype: None = None, level: ThreadLevel = "thread") -> Any:
        """Omit dtype or use an ``Any``-typed external CUTLASS dtype token."""

    def sync(self) -> None:
        """Synchronize participating members.

        Grid synchronization requires a compiler-verified cooperative launch.
        """

    def sync_aligned(self) -> None:
        """Synchronize an aligned group in converged control flow.

        Grid synchronization requires a compiler-verified cooperative launch.
        """

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

    def is_member(self) -> CompilerIntegerLike:
        """Return a CUTLASS ``Uint8`` membership flag for the current thread."""

MemoryGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp", "block"]]
MergeSortWarpGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp"]]
ReductionGroup: TypeAlias = ThreadGroup[
    Literal[
        "thread",
        "warp",
        "threads_within_warp",
        "warps_within_block",
        "block",
        "cluster",
    ]
]
BlockGroup: TypeAlias = ThreadGroup[Literal["block"]]
WarpGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp"]]

def this_thread() -> ThreadGroup[Literal["thread"]]:
    """Describe the current thread."""

def this_warp() -> ThreadGroup[Literal["warp"]]:
    """Describe the current complete physical warp."""

def this_block() -> ThreadGroup[Literal["block"]]:
    """Describe the current CUDA thread block."""

def this_cluster() -> ThreadGroup[Literal["cluster"]]:
    """Describe the current thread-block cluster."""

def this_grid() -> ThreadGroup[Literal["grid"]]:
    """Describe the current grid."""

__all__ = [
    "Hierarchy",
    "ThreadGroup",
    "ThreadHierarchy",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]
