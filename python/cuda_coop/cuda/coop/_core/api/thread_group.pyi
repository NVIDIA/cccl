# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable CUDA thread groups and hierarchy."""

from typing import Callable, Generic, Literal, TypeAlias, overload

from typing_extensions import Self, TypeVar

from cuda.coop._typing import (
    IntegerValue,
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

class ThreadHierarchy:
    """CUDA hierarchy descriptor resolved from the current kernel launch."""

    block_dim: tuple[int, int, int] | None
    grid_dim: tuple[int, int, int] | None
    cluster_dim: tuple[int, int, int] | None
    implicit: bool

    def __init__(self) -> None:
        """Describe the compiler's current launch hierarchy."""

    @classmethod
    def current(cls) -> Self:
        """Describe the compiler's current launch hierarchy."""

    @property
    def is_static(self) -> bool:
        """Return whether the hierarchy carries static dimensions."""

    @property
    def block_thread_count(self) -> int | None:
        """Return the statically known CTA size, if any."""

Hierarchy: TypeAlias = ThreadHierarchy

class ThreadGroup(Generic[_GroupKindT_co]):
    """Common compiler-facing contract for one CUDA thread group."""

    hierarchy: ThreadHierarchy
    block_dim: tuple[int, int, int] | None
    parent: ThreadGroup | None
    source: str

    @property
    def kind(self) -> _GroupKindT_co:
        """Return the kind carried by this group descriptor."""

    @property
    def static_size(self) -> int | None:
        """Return the statically known group extent, if any."""

    @property
    def group_thread_count(self) -> int:
        """Return the group extent, requiring it to be statically known."""

    @property
    def is_current(self) -> bool:
        """Return whether this descriptor refers to the current hierarchy."""

    @property
    def is_static(self) -> bool:
        """Return whether dimensions needed by this group are static."""

    @overload
    def rank(self, level: _UniversalQueryLevel = "thread") -> IntegerValue:
        """Query a universal thread or Warp level on an unnarrowed group."""
    @overload
    def rank(
        self: ThreadGroup[_PhysicalGroupKind], level: ThreadLevel = "thread"
    ) -> IntegerValue:
        """Return rank using the outer C++ hierarchy boundary's product type."""
    @overload
    def rank(
        self: ThreadGroup[Literal["threads_within_warp"]],
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> IntegerValue:
        """Query a logical Warp's constituent threads or parent Warp."""
    @overload
    def rank(
        self: ThreadGroup[Literal["warps_within_block"]],
        level: _WarpsWithinBlockLevel = "thread",
    ) -> IntegerValue:
        """Query a mapped group's constituents or parent block."""

    @overload
    def count(self, level: _UniversalQueryLevel = "thread") -> IntegerValue:
        """Query a universal thread or Warp level on an unnarrowed group."""
    @overload
    def count(
        self: ThreadGroup[_PhysicalGroupKind], level: ThreadLevel = "thread"
    ) -> IntegerValue:
        """Return count using the outer C++ hierarchy boundary's product type."""
    @overload
    def count(
        self: ThreadGroup[Literal["threads_within_warp"]],
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> IntegerValue:
        """Query a logical Warp's constituent threads or parent Warp."""
    @overload
    def count(
        self: ThreadGroup[Literal["warps_within_block"]],
        level: _WarpsWithinBlockLevel = "thread",
    ) -> IntegerValue:
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
    ) -> IntegerValue:
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
        """Return rank converted to an integral dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: type[_ItemT],
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> _ItemT:
        """Return rank converted to an integral dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: type[_ItemT],
        level: _WarpsWithinBlockLevel = "thread",
    ) -> _ItemT:
        """Return rank converted to an integral dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[_PhysicalGroupKind],
        dtype: None = None,
        level: ThreadLevel = "thread",
    ) -> IntegerValue:
        """Use the C++ hierarchy operation's default unsigned dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: None = None,
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> IntegerValue:
        """Use the C++ hierarchy operation's default unsigned dtype."""
    @overload
    def rank_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: None = None,
        level: _WarpsWithinBlockLevel = "thread",
    ) -> IntegerValue:
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
    ) -> IntegerValue:
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
        """Return count converted to an integral dtype."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: type[_ItemT],
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> _ItemT:
        """Return count converted to an integral dtype."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: type[_ItemT],
        level: _WarpsWithinBlockLevel = "thread",
    ) -> _ItemT:
        """Return count converted to an integral dtype."""
    @overload
    def count_as(
        self: ThreadGroup[_PhysicalGroupKind],
        dtype: None = None,
        level: ThreadLevel = "thread",
    ) -> IntegerValue:
        """Use the C++ hierarchy operation's default unsigned dtype."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["threads_within_warp"]],
        dtype: None = None,
        level: _ThreadsWithinWarpLevel = "thread",
    ) -> IntegerValue:
        """Use the C++ hierarchy operation's default unsigned dtype."""
    @overload
    def count_as(
        self: ThreadGroup[Literal["warps_within_block"]],
        dtype: None = None,
        level: _WarpsWithinBlockLevel = "thread",
    ) -> IntegerValue:
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

    def is_member(self) -> IntegerValue:
        """Return whether the current thread belongs to this group."""

MemoryGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp", "block"]]
ReductionGroup: TypeAlias = ThreadGroup[
    Literal[
        "thread",
        "warp",
        "threads_within_warp",
        "block",
        "warps_within_block",
        "cluster",
    ]
]
BlockGroup: TypeAlias = ThreadGroup[Literal["block"]]
WarpGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp"]]

def this_thread() -> ThreadGroup[Literal["thread"]]:
    """Describe the current thread."""

def this_warp() -> ThreadGroup[Literal["warp"]]:
    """Describe the current physical warp."""

def this_block() -> ThreadGroup[Literal["block"]]:
    """Describe the current CUDA thread block."""

def this_cluster() -> ThreadGroup[Literal["cluster"]]:
    """Describe the current thread-block cluster."""

def this_grid() -> ThreadGroup[Literal["grid"]]:
    """Describe the current grid; grid collectives are not in the portable API."""
