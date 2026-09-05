# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable CUDA thread-block descriptor."""

from typing import Literal

class _ThreadGroupConstructionToken: ...

class ThreadGroup:
    """Descriptor for the current CUDA thread block.

    The descriptor is compiler-free. A backend resolves its exact dimensions
    from verified launch facts while tracing a cooperative reduction.

    Raises:
        TypeError: If user code attempts to construct the opaque descriptor
            directly instead of calling ``this_block``.

    Example:
        >>> from cuda import coop
        >>> block = coop.this_block()
        >>> block.kind
        'block'
    """

    def __init__(self, _token: _ThreadGroupConstructionToken, /) -> None: ...
    @property
    def kind(self) -> Literal["block"]: ...

def this_block() -> ThreadGroup:
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

__all__ = ["ThreadGroup", "this_block"]
