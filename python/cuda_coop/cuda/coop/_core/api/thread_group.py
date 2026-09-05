# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable constructors for CUDA thread groups.

This module owns the public constructor surface. Static hierarchy resolution
and group identity remain in the backend-neutral thread-group model.
"""

from __future__ import annotations

from ..thread_group import ThreadGroup
from ..thread_group import this_block as _core_this_block


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


__all__ = ["ThreadGroup", "this_block"]
