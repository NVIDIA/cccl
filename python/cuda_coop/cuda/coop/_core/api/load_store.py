# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiler-recognized calls for cooperative load and store.

The compiler checks these arguments and generates the device operation.
The Python bodies reject calls outside a supported kernel.
"""

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup
from ._dispatch import (
    _portable_group_operation,
)
from ._payload import (
    TempStorageLike,
    ThreadDataLike,
)


@_portable_group_operation(
    "load",
    group_kinds=("block", "warp", "threads_within_warp"),
)
def load(
    group: ThreadGroup,
    source: object,
    output: ThreadDataLike[Any],
    /,
    *,
    algorithm: str = "direct",
    valid_items: object = None,
    oob_default: object = None,
    offset: object = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Populate ``output`` cooperatively in place and return ``None``.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.load must be called from a supported GPU kernel."
    )


@_portable_group_operation(
    "store",
    group_kinds=("block", "warp", "threads_within_warp"),
)
def store(
    group: ThreadGroup,
    destination: object,
    value: object,
    /,
    *,
    algorithm: str = "direct",
    valid_items: object = None,
    offset: object = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Store values cooperatively through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.store must be called from a supported GPU kernel."
    )


__all__ = ["load", "store"]
