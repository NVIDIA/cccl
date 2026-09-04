# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable CUDA thread groups and hierarchy."""

from typing import Generic, Literal, TypeAlias, overload

from typing_extensions import Self, TypeVar

from cuda.coop._typing import ThreadGroupKind

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
