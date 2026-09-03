# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Thread-group descriptors and constructors for the qualified backend."""

from typing import Any, Generic, Literal, TypeAlias, overload

import numpy as np
from typing_extensions import TypeVar

from .. import ThreadHierarchy
from .._core.api.thread_group import ThreadGroup as PortableThreadGroup
from .._typing import SynchronizableGroupKind, ThreadGroupKind, ThreadLevel

_ItemT = TypeVar("_ItemT")

_GroupKindT_co = TypeVar(
    "_GroupKindT_co",
    bound=ThreadGroupKind,
    covariant=True,
    default=ThreadGroupKind,
)

Hierarchy = ThreadHierarchy

class ThreadGroup(
    PortableThreadGroup[_GroupKindT_co],
    Generic[_GroupKindT_co],
):
    """Compile-time CUDA group descriptor for Numba-CUDA-MLIR."""

    def rank(self, level: ThreadLevel = "thread") -> np.int32:
        """Return this group's rank as a NumPy-compatible ``int32`` scalar."""

    def count(self, level: ThreadLevel = "thread") -> np.int32:
        """Return this group's count as a NumPy-compatible ``int32`` scalar."""

    @overload
    def rank_as(self, dtype: type[_ItemT], level: ThreadLevel = "thread") -> _ItemT:
        """Return the group rank converted to an ordinary scalar dtype."""

    @overload
    def rank_as(self, dtype: object = None, level: ThreadLevel = "thread") -> Any:
        """Return the group rank converted to a compiler dtype token."""

    @overload
    def count_as(
        self,
        dtype: type[_ItemT],
        level: ThreadLevel = "thread",
    ) -> _ItemT:
        """Return the group count converted to an ordinary scalar dtype."""

    @overload
    def count_as(self, dtype: object = None, level: ThreadLevel = "thread") -> Any:
        """Return the group count converted to a compiler dtype token."""

    def sync(self: ThreadGroup[SynchronizableGroupKind]) -> None:
        """Synchronize participating members; grid groups are unsupported."""

    def sync_aligned(self: ThreadGroup[SynchronizableGroupKind]) -> None:
        """Synchronize a converged non-grid group."""

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

    def is_member(self) -> np.uint8:
        """Return a NumPy-compatible ``uint8`` membership flag."""

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
    """Describe the current cluster where the launch can represent it."""

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
