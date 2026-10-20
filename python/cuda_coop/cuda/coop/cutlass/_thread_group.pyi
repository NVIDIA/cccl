# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declare CUTLASS groups and their CuTe query result types.

Overloads restrict mapped queries to their constituents and immediate parent.
Python or NumPy dtype selectors choose a scalar representation; runtime
values still belong to CuTe. Default widths are Uint64 for grid-related
queries and Uint32 otherwise.
"""

from collections.abc import Callable
from typing import Generic, Literal, TypeAlias, overload

from cutlass import Uint8, Uint32, Uint64
from typing_extensions import TypeVar

from .. import ThreadHierarchy
from .._core.api.thread_group import ThreadGroup as CommonThreadGroup
from .._typing import (
    SynchronizableGroupKind,
    ThreadGroupKind,
    ThreadGroupQueryScalar,
    ThreadLevel,
)

_ItemT = TypeVar("_ItemT", bound=ThreadGroupQueryScalar)
_BuiltinIntDType: TypeAlias = Callable[[str | bytes | bytearray, int], int]
_PhysicalGroupKind: TypeAlias = Literal[
    "thread", "warp", "block", "cluster", "grid"
]
_UniversalQueryLevel: TypeAlias = Literal["thread", "gpu_thread", "warp"]
_ThreadsWithinWarpLevel: TypeAlias = Literal["thread", "gpu_thread", "warp"]
_WarpsWithinBlockLevel: TypeAlias = Literal[
    "thread", "gpu_thread", "warp", "block"
]
_GroupKindT_co = TypeVar(
    "_GroupKindT_co",
    bound=ThreadGroupKind,
    covariant=True,
    default=ThreadGroupKind,
)

Hierarchy: TypeAlias = ThreadHierarchy

class ThreadGroup(
    CommonThreadGroup[_GroupKindT_co],
    Generic[_GroupKindT_co],
):
    """Compile-time CUDA group descriptor for CUTLASS."""

    @overload
    def rank(self, level: _UniversalQueryLevel = "thread") -> Uint32 | Uint64:
        """Query a universal thread or Warp level on an unnarrowed group."""
    @overload
    def rank(
        self: ThreadGroup[_PhysicalGroupKind], level: ThreadLevel = "thread"
    ) -> Uint32 | Uint64:
        """Return an unsigned rank; grid-related queries use 64 bits."""
    @overload
    def rank(
        self: ThreadGroup[Literal["threads_within_warp"]],
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> Uint32 | Uint64:
        """Query a logical Warp's constituent threads or parent Warp."""
    @overload
    def rank(
        self: ThreadGroup[Literal["warps_within_block"]],
        level: _WarpsWithinBlockLevel = "thread",
    ) -> Uint32 | Uint64:
        """Query a mapped group's constituents or parent block."""

    @overload
    def count(self, level: _UniversalQueryLevel = "thread") -> Uint32 | Uint64:
        """Query a universal thread or Warp level on an unnarrowed group."""
    @overload
    def count(
        self: ThreadGroup[_PhysicalGroupKind], level: ThreadLevel = "thread"
    ) -> Uint32 | Uint64:
        """Return an unsigned count; grid-related queries use 64 bits."""
    @overload
    def count(
        self: ThreadGroup[Literal["threads_within_warp"]],
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> Uint32 | Uint64:
        """Query a logical Warp's constituent threads or parent Warp."""
    @overload
    def count(
        self: ThreadGroup[Literal["warps_within_block"]],
        level: _WarpsWithinBlockLevel = "thread",
    ) -> Uint32 | Uint64:
        """Query a mapped group's constituents or parent block."""

    @overload
    def rank_as(
        self,
        dtype: _BuiltinIntDType,
        level: _UniversalQueryLevel = "thread",
    ) -> int:
        """Use ``int`` to select a signed 32-bit CuTe ``Int32`` rank."""
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
    ) -> Uint32 | Uint64:
        """Query a universal thread or Warp rank with the default dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[_PhysicalGroupKind],
        dtype: _BuiltinIntDType,
        level: ThreadLevel = "thread",
    ) -> int:
        """Use ``int`` to select a signed 32-bit CuTe ``Int32`` rank."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: _BuiltinIntDType,
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> int:
        """Use ``int`` to select a signed 32-bit CuTe ``Int32`` rank."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: _BuiltinIntDType,
        level: _WarpsWithinBlockLevel = "thread",
    ) -> int:
        """Use ``int`` to select a signed 32-bit CuTe ``Int32`` rank."""
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
    ) -> Uint32 | Uint64:
        """Use the C++ hierarchy operation's default unsigned dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: None = None,
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> Uint32 | Uint64:
        """Use the C++ hierarchy operation's default unsigned dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: None = None,
        level: _WarpsWithinBlockLevel = "thread",
    ) -> Uint32 | Uint64:
        """Use the C++ hierarchy operation's default unsigned dtype."""

    @overload
    def count_as(
        self,
        dtype: _BuiltinIntDType,
        level: _UniversalQueryLevel = "thread",
    ) -> int:
        """Use ``int`` to select a signed 32-bit CuTe ``Int32`` count."""
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
    ) -> Uint32 | Uint64:
        """Query a universal thread or Warp count with the default dtype."""
    @overload
    def count_as(
        self: ThreadGroup[_PhysicalGroupKind],
        dtype: _BuiltinIntDType,
        level: ThreadLevel = "thread",
    ) -> int:
        """Use ``int`` to select a signed 32-bit CuTe ``Int32`` count."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: _BuiltinIntDType,
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> int:
        """Use ``int`` to select a signed 32-bit CuTe ``Int32`` count."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: _BuiltinIntDType,
        level: _WarpsWithinBlockLevel = "thread",
    ) -> int:
        """Use ``int`` to select a signed 32-bit CuTe ``Int32`` count."""
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
    ) -> Uint32 | Uint64:
        """Use the C++ hierarchy operation's default unsigned dtype."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: None = None,
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> Uint32 | Uint64:
        """Use the C++ hierarchy operation's default unsigned dtype."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: None = None,
        level: _WarpsWithinBlockLevel = "thread",
    ) -> Uint32 | Uint64:
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
        """Partition a physical warp into groups of ``count`` threads.

        ``count`` is a compile-time value from 1 to 32. With
        ``exhaustive=False``, a count that does not divide 32 leaves trailing
        lanes outside every group. Each primitive documents its supported
        widths; Load/Store accept 1, 2, 4, 8, 16, and 32.
        """

    @overload
    def group_by(
        self: ThreadGroup[Literal["block"]],
        count: int,
        *,
        exhaustive: bool = True,
    ) -> ThreadGroup[Literal["warps_within_block"]]:
        """Partition a block into groups of physical warps."""

    def is_member(self) -> Uint8:
        """Return a CuTe ``Uint8`` membership flag."""

BlockGroup: TypeAlias = ThreadGroup[Literal["block"]]

WarpGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp"]]
MemoryGroup: TypeAlias = BlockGroup | WarpGroup

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
