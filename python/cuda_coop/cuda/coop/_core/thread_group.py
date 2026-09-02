# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUDA thread-group descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

Dim3 = tuple[int, int, int]
PHYSICAL_WARP_THREADS = 32
ThreadGroupKind = Literal["block", "warp"]
_GroupKindT_co = TypeVar(
    "_GroupKindT_co",
    bound=ThreadGroupKind,
    covariant=True,
)


def normalize_thread_dim(
    value: Any,
    *,
    scope: str,
    label: str,
) -> Dim3:
    """Normalize one positive one-, two-, or three-dimensional extent."""

    if isinstance(value, int) and not isinstance(value, bool):
        dimensions = (value,)
    elif isinstance(value, (tuple, list)):
        dimensions = tuple(value)
    else:
        raise TypeError(f"{scope} {label} dimensions must be an integer or sequence")
    if not 1 <= len(dimensions) <= 3:
        raise ValueError(f"{scope} {label} dimensions must have one to three axes")
    if any(
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
        for dimension in dimensions
    ):
        raise ValueError(f"{scope} {label} dimensions must be positive integers")
    return (*dimensions, *(1 for _ in range(3 - len(dimensions))))


@dataclass(frozen=True)
class ThreadHierarchy:
    """Static hierarchy facts attached after compiler launch resolution."""

    block_dim: Dim3 | None = None

    def __post_init__(self) -> None:
        if self.block_dim is not None:
            object.__setattr__(
                self,
                "block_dim",
                normalize_thread_dim(
                    self.block_dim,
                    scope="ThreadHierarchy",
                    label="block",
                ),
            )

    @property
    def block_thread_count(self) -> int | None:
        if self.block_dim is None:
            return None
        x, y, z = self.block_dim
        return x * y * z


@dataclass(frozen=True, init=False)
class ThreadGroup(Generic[_GroupKindT_co]):
    """Descriptor for the current CUDA thread block or physical warp.

    The descriptor is compiler-free. A backend resolves its exact dimensions
    from verified launch facts while tracing a cooperative operation.

    Raises:
        TypeError: If user code attempts to construct the opaque descriptor
            directly instead of calling ``this_block`` or ``this_warp``.

    Example:
        >>> from cuda import coop
        >>> block = coop.this_block()
        >>> block.kind
        'block'
    """

    kind: ThreadGroupKind = "block"
    hierarchy: ThreadHierarchy = field(default_factory=ThreadHierarchy)
    source: str = field(default="current", compare=False, hash=False)

    def __init__(self) -> None:
        raise TypeError(
            "ThreadGroup descriptors are opaque; call cuda.coop.this_block() "
            "or cuda.coop.this_warp()"
        )

    @classmethod
    def _create(
        cls,
        *,
        kind: ThreadGroupKind,
        hierarchy: ThreadHierarchy | None = None,
        source: str = "current",
    ) -> ThreadGroup:
        if kind not in {"block", "warp"}:
            raise ValueError(f"unsupported thread group kind {kind!r}")
        result = object.__new__(cls)
        object.__setattr__(result, "kind", kind)
        object.__setattr__(result, "hierarchy", hierarchy or ThreadHierarchy())
        object.__setattr__(result, "source", source)
        return result

    @property
    def is_current(self) -> bool:
        return self.hierarchy.block_dim is None

    @property
    def static_size(self) -> int | None:
        if self.kind == "warp":
            return PHYSICAL_WARP_THREADS
        return self.hierarchy.block_thread_count

    @property
    def semantic_key(self) -> tuple[str, Dim3 | None]:
        return self.kind, self.hierarchy.block_dim

    def with_hierarchy(
        self,
        hierarchy: ThreadHierarchy,
        *,
        source: str,
    ) -> ThreadGroup:
        if not isinstance(hierarchy, ThreadHierarchy):
            raise TypeError("ThreadGroup hierarchy must be a ThreadHierarchy")
        return ThreadGroup._create(
            kind=self.kind,
            hierarchy=hierarchy,
            source=source,
        )


def this_block() -> ThreadGroup:
    """Return a descriptor for the current CUDA thread block.

    The returned group has no user-supplied dimensions. The active compiler
    backend supplies exact launch facts when it lowers a reduction.

    Returns:
        A compiler-free block descriptor accepted by cooperative primitives.

    Raises:
        RuntimeError: If a compiler backend later cannot resolve exact block
            dimensions for an operation using this descriptor.

    Example:
        >>> from cuda import coop
        >>> block = coop.this_block()
        >>> block.kind
        'block'
    """

    return ThreadGroup._create(kind="block")


def this_warp() -> ThreadGroup:
    """Return a descriptor for the current physical CUDA warp.

    The descriptor always represents 32 lanes. A compiler integration resolves
    the surrounding block dimensions and rejects launches containing a partial
    physical warp when lowering a collective.

    Returns:
        A compiler-free physical-warp descriptor accepted by cooperative
        primitives.

    Raises:
        RuntimeError: If a compiler later cannot resolve exact compatible block
            dimensions.

    Example:
        >>> from cuda import coop
        >>> warp = coop.this_warp()
        >>> warp.kind
        'warp'
    """

    return ThreadGroup._create(kind="warp")


__all__ = [
    "Dim3",
    "PHYSICAL_WARP_THREADS",
    "ThreadGroup",
    "ThreadGroupKind",
    "ThreadHierarchy",
    "normalize_thread_dim",
    "this_block",
    "this_warp",
]
