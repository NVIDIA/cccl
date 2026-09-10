# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Thread-group descriptors and constructors for the qualified backend."""

from typing import Callable, Generic, Literal, TypeAlias, overload

import numpy as np
from typing_extensions import TypeVar

from .. import ThreadHierarchy
from .._core.api.thread_group import ThreadGroup as PortableThreadGroup
from .._typing import (
    SynchronizableGroupKind,
    ThreadGroupKind,
    ThreadGroupQueryScalar,
    ThreadLevel,
)

_ItemT = TypeVar("_ItemT", bound=ThreadGroupQueryScalar)
_BuiltinIntDType: TypeAlias = Callable[[str | bytes | bytearray, int], int]
_PhysicalGroupKind: TypeAlias = Literal["thread", "warp", "block", "cluster", "grid"]
_UniversalQueryLevel: TypeAlias = Literal["thread", "gpu_thread", "warp"]
_ThreadsWithinWarpLevel: TypeAlias = Literal["thread", "gpu_thread", "warp"]
_WarpsWithinBlockLevel: TypeAlias = Literal["thread", "gpu_thread", "warp", "block"]
_GroupKindT_co = TypeVar(
    "_GroupKindT_co",
    bound=ThreadGroupKind,
    covariant=True,
    default=ThreadGroupKind,
)

Hierarchy: TypeAlias = ThreadHierarchy

class ThreadGroup(
    PortableThreadGroup[_GroupKindT_co],
    Generic[_GroupKindT_co],
):
    """Compile-time CUDA group descriptor for Numba-CUDA-MLIR."""

    @overload
    def rank(self, level: _UniversalQueryLevel = "thread") -> np.uint32 | np.uint64:
        """Query a universal thread or Warp level on an unnarrowed group."""
    @overload
    def rank(
        self: ThreadGroup[_PhysicalGroupKind], level: ThreadLevel = "thread"
    ) -> np.uint32 | np.uint64:
        """Return rank using the outer C++ hierarchy boundary's product type."""
    @overload
    def rank(
        self: ThreadGroup[Literal["threads_within_warp"]],
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> np.uint32 | np.uint64:
        """Query a logical Warp's constituent threads or parent Warp."""
    @overload
    def rank(
        self: ThreadGroup[Literal["warps_within_block"]],
        level: _WarpsWithinBlockLevel = "thread",
    ) -> np.uint32 | np.uint64:
        """Query a mapped group's constituents or parent block."""

    @overload
    def count(self, level: _UniversalQueryLevel = "thread") -> np.uint32 | np.uint64:
        """Query a universal thread or Warp level on an unnarrowed group."""
    @overload
    def count(
        self: ThreadGroup[_PhysicalGroupKind], level: ThreadLevel = "thread"
    ) -> np.uint32 | np.uint64:
        """Return count using the outer C++ hierarchy boundary's product type."""
    @overload
    def count(
        self: ThreadGroup[Literal["threads_within_warp"]],
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> np.uint32 | np.uint64:
        """Query a logical Warp's constituent threads or parent Warp."""
    @overload
    def count(
        self: ThreadGroup[Literal["warps_within_block"]],
        level: _WarpsWithinBlockLevel = "thread",
    ) -> np.uint32 | np.uint64:
        """Query a mapped group's constituents or parent block."""

    @overload
    def rank_as(
        self,
        dtype: _BuiltinIntDType,
        level: _UniversalQueryLevel = "thread",
    ) -> int:
        """Query a universal thread or Warp rank as built-in int."""
    @overload
    def rank_as(
        self,
        dtype: type[_ItemT],
        level: _UniversalQueryLevel = "thread",
    ) -> _ItemT:
        """Query a universal thread or Warp rank as an integer dtype."""
    @overload
    def rank_as(
        self,
        dtype: None = None,
        level: _UniversalQueryLevel = "thread",
    ) -> np.uint32 | np.uint64:
        """Query a universal thread or Warp rank with the default dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[_PhysicalGroupKind],
        dtype: _BuiltinIntDType,
        level: ThreadLevel = "thread",
    ) -> int:
        """Return rank converted to the built-in integral dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: _BuiltinIntDType,
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> int:
        """Return rank converted to the built-in integral dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: _BuiltinIntDType,
        level: _WarpsWithinBlockLevel = "thread",
    ) -> int:
        """Return rank converted to the built-in integral dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[_PhysicalGroupKind],
        dtype: type[_ItemT],
        level: ThreadLevel = "thread",
    ) -> _ItemT:
        """Return the group rank converted to an integral dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: type[_ItemT],
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> _ItemT:
        """Return the group rank converted to an integral dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: type[_ItemT],
        level: _WarpsWithinBlockLevel = "thread",
    ) -> _ItemT:
        """Return the group rank converted to an integral dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[_PhysicalGroupKind],
        dtype: None = None,
        level: ThreadLevel = "thread",
    ) -> np.uint32 | np.uint64:
        """Use the C++ hierarchy operation's default unsigned dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: None = None,
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> np.uint32 | np.uint64:
        """Use the C++ hierarchy operation's default unsigned dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: None = None,
        level: _WarpsWithinBlockLevel = "thread",
    ) -> np.uint32 | np.uint64:
        """Use the C++ hierarchy operation's default unsigned dtype."""

    @overload
    def count_as(
        self,
        dtype: _BuiltinIntDType,
        level: _UniversalQueryLevel = "thread",
    ) -> int:
        """Query a universal thread or Warp count as built-in int."""
    @overload
    def count_as(
        self,
        dtype: type[_ItemT],
        level: _UniversalQueryLevel = "thread",
    ) -> _ItemT:
        """Query a universal thread or Warp count as an integer dtype."""
    @overload
    def count_as(
        self,
        dtype: None = None,
        level: _UniversalQueryLevel = "thread",
    ) -> np.uint32 | np.uint64:
        """Query a universal thread or Warp count with the default dtype."""
    @overload
    def count_as(
        self: ThreadGroup[_PhysicalGroupKind],
        dtype: _BuiltinIntDType,
        level: ThreadLevel = "thread",
    ) -> int:
        """Return count converted to the built-in integral dtype."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: _BuiltinIntDType,
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> int:
        """Return count converted to the built-in integral dtype."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: _BuiltinIntDType,
        level: _WarpsWithinBlockLevel = "thread",
    ) -> int:
        """Return count converted to the built-in integral dtype."""
    @overload
    def count_as(
        self: ThreadGroup[_PhysicalGroupKind],
        dtype: type[_ItemT],
        level: ThreadLevel = "thread",
    ) -> _ItemT:
        """Return the group count converted to an integral dtype."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: type[_ItemT],
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> _ItemT:
        """Return the group count converted to an integral dtype."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: type[_ItemT],
        level: _WarpsWithinBlockLevel = "thread",
    ) -> _ItemT:
        """Return the group count converted to an integral dtype."""
    @overload
    def count_as(
        self: ThreadGroup[_PhysicalGroupKind],
        dtype: None = None,
        level: ThreadLevel = "thread",
    ) -> np.uint32 | np.uint64:
        """Use the C++ hierarchy operation's default unsigned dtype."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: None = None,
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> np.uint32 | np.uint64:
        """Use the C++ hierarchy operation's default unsigned dtype."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: None = None,
        level: _WarpsWithinBlockLevel = "thread",
    ) -> np.uint32 | np.uint64:
        """Use the C++ hierarchy operation's default unsigned dtype."""

    def sync(self: ThreadGroup[SynchronizableGroupKind]) -> None:
        """Synchronize members of a non-grid, non-mapped-warp group."""

    def sync_aligned(self: ThreadGroup[SynchronizableGroupKind]) -> None:
        """Synchronize an aligned, converged group with supported barriers."""

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
