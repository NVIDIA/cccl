# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable CUDA thread groups and hierarchy."""

from typing import Any, Generic, Literal, TypeAlias, overload

from typing_extensions import Self, TypeVar

from cuda.coop._typing import ThreadGroupKind as _ThreadGroupKind
from cuda.coop._typing import ThreadLevel as _ThreadLevel
from cuda.coop._typing import _IntegerValue as _IntegerValue
from cuda.coop._typing import _PortableNumericScalar as _PortableNumericScalar
from cuda.coop._typing import _SynchronizableGroupKind as _SynchronizableGroupKind

_ItemT = TypeVar("_ItemT", bound=_PortableNumericScalar)
_GroupKindT_co = TypeVar(
    "_GroupKindT_co",
    bound=_ThreadGroupKind,
    covariant=True,
    default=_ThreadGroupKind,
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
    def is_current(self) -> bool:
        """Return whether this descriptor refers to the current hierarchy."""

    @property
    def is_static(self) -> bool:
        """Return whether dimensions needed by this group are static."""

    def rank(self, level: _ThreadLevel = "thread") -> _IntegerValue:
        """Return this group's rank relative to another hierarchy level.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.rank requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """

    def count(self, level: _ThreadLevel = "thread") -> _IntegerValue:
        """Return this group's count relative to another hierarchy level.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.count requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """

    @overload
    def rank_as(self, dtype: type[_ItemT], level: _ThreadLevel = "thread") -> _ItemT:
        """Return rank converted to a dtype.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.rank_as requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """
    @overload
    def rank_as(self, dtype: None = None, level: _ThreadLevel = "thread") -> Any:
        """Omit dtype or use an ``Any``-typed external compiler dtype token.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.rank_as requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """

    @overload
    def count_as(self, dtype: type[_ItemT], level: _ThreadLevel = "thread") -> _ItemT:
        """Return count converted to a dtype.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.count_as requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """
    @overload
    def count_as(self, dtype: None = None, level: _ThreadLevel = "thread") -> Any:
        """Omit dtype or use an ``Any``-typed external compiler dtype token.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.count_as requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """

    def sync(self: ThreadGroup[_SynchronizableGroupKind]) -> None:
        """Synchronize the group's members.

        Grid synchronization is unavailable through the portable API.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.sync requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """

    def sync_aligned(self: ThreadGroup[_SynchronizableGroupKind]) -> None:
        """Synchronize an aligned group.

        Grid synchronization is unavailable through the portable API.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.sync_aligned requires a Python DSL compiler
        context (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
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
    def is_member(self) -> _IntegerValue:
        """Return whether the current thread belongs to this group.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.is_member requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """

_MemoryGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp", "block"]]
_ReductionGroup: TypeAlias = ThreadGroup[
    Literal["thread", "warp", "threads_within_warp", "block", "cluster"]
]
_BlockGroup: TypeAlias = ThreadGroup[Literal["block"]]
_WarpGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp"]]

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
