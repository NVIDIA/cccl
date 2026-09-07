# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUDA thread-block descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Dim3 = tuple[int, int, int]


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
class ThreadGroup:
    """Descriptor for the current CUDA thread block.

    Creating this descriptor does not require an active compiler backend. When
    a cooperative primitive is compiled, the backend resolves the descriptor
    against the kernel's exact launch dimensions.

    Raises:
        TypeError: If user code attempts to construct the opaque descriptor
            directly instead of calling ``this_block``.

    Example:
        This tested CUTLASS kernel uses the current CUDA thread block:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """

    kind: Literal["block"] = "block"
    hierarchy: ThreadHierarchy = field(default_factory=ThreadHierarchy)
    source: str = field(default="current", compare=False, hash=False)

    def __init__(self) -> None:
        raise TypeError(
            "ThreadGroup descriptors are opaque; call cuda.coop.this_block()"
        )

    @classmethod
    def _create(
        cls,
        *,
        hierarchy: ThreadHierarchy | None = None,
        source: str = "current",
    ) -> ThreadGroup:
        result = object.__new__(cls)
        object.__setattr__(result, "kind", "block")
        object.__setattr__(result, "hierarchy", hierarchy or ThreadHierarchy())
        object.__setattr__(result, "source", source)
        return result

    @property
    def is_current(self) -> bool:
        return self.hierarchy.block_dim is None

    @property
    def static_size(self) -> int | None:
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
        return ThreadGroup._create(hierarchy=hierarchy, source=source)


def this_block() -> ThreadGroup:
    """Return a descriptor for the current CUDA thread block.

    The returned group has no user-supplied dimensions. The active backend
    supplies exact launch dimensions when it lowers a cooperative primitive.

    Returns:
        An opaque block descriptor accepted by cooperative primitives.

    Raises:
        RuntimeError: If a compiler backend later cannot resolve exact block
            dimensions for an operation using this descriptor.

    Example:
        This tested CUTLASS kernel uses the current CUDA thread block:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """

    return ThreadGroup._create()


__all__ = [
    "Dim3",
    "ThreadGroup",
    "ThreadHierarchy",
    "normalize_thread_dim",
    "this_block",
]
