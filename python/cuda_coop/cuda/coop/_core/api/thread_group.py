# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable constructors for CUDA thread groups.

This module owns the public constructor surface. Static hierarchy resolution
and group identity remain in the backend-neutral thread-group model.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from ..thread_group import ThreadGroup, ThreadGroupKind
from ..thread_group import this_block as _core_this_block
from ..thread_group import this_warp as _core_this_warp

BlockGroup: TypeAlias = ThreadGroup[Literal["block"]]
WarpGroup: TypeAlias = ThreadGroup[Literal["warp"]]


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

    return _core_this_block()


def this_warp() -> ThreadGroup:
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

    return _core_this_warp()


__all__ = [
    "BlockGroup",
    "ThreadGroup",
    "ThreadGroupKind",
    "WarpGroup",
    "this_block",
    "this_warp",
]
