# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable CUDA thread-group descriptors."""

from typing import Generic, Literal, TypeAlias, TypeVar

ThreadGroupKind: TypeAlias = Literal["block", "warp"]
_GroupKindT_co = TypeVar(
    "_GroupKindT_co",
    bound=ThreadGroupKind,
    covariant=True,
)

class _ThreadGroupConstructionToken: ...

class ThreadGroup(Generic[_GroupKindT_co]):
    """Descriptor for the current CUDA thread block or physical warp.

    The descriptor is compiler-free. A backend resolves its exact dimensions
    from verified launch facts while tracing a cooperative reduction.

    Raises:
        TypeError: If user code attempts to construct the opaque descriptor
            directly instead of calling ``this_block`` or ``this_warp``.

    Example:
        >>> from cuda import coop
        >>> block = coop.this_block()
        >>> block.kind
        'block'
    """

    def __init__(self, _token: _ThreadGroupConstructionToken, /) -> None: ...
    @property
    def kind(self) -> _GroupKindT_co: ...
    @property
    def static_size(self) -> int | None: ...

BlockGroup: TypeAlias = ThreadGroup[Literal["block"]]
WarpGroup: TypeAlias = ThreadGroup[Literal["warp"]]

def this_block() -> BlockGroup:
    """Return a descriptor for the current CUDA thread block.

    The descriptor carries no user-supplied dimensions. A compiler integration
    resolves exact block dimensions from verified launch facts when lowering a
    reduction.

    Returns:
        A compiler-free block descriptor accepted by ``reduce`` and ``sum``.

    Raises:
        RuntimeError: If a compiler later cannot resolve exact block dimensions.

    Example:
        >>> from cuda import coop
        >>> block = coop.this_block()
        >>> block.kind
        'block'
    """

def this_warp() -> WarpGroup:
    """Return a descriptor for the current physical CUDA warp.

    The descriptor always represents the 32 lanes in the calling thread's
    physical warp. A compiler integration resolves the enclosing block shape
    and rejects launches containing a partial warp.

    Returns:
        A compiler-free physical-warp descriptor accepted by ``reduce`` and
        ``sum``.

    Raises:
        RuntimeError: If a compiler later cannot resolve exact compatible block
            dimensions.

    Example:
        >>> from cuda import coop
        >>> warp = coop.this_warp()
        >>> warp.static_size
        32
    """

__all__ = [
    "BlockGroup",
    "ThreadGroup",
    "ThreadGroupKind",
    "WarpGroup",
    "this_block",
    "this_warp",
]
