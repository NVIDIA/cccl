# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative Shuffle entry point."""

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup
from ._dispatch import (
    _portable_group_operation,
)
from ._payload import (
    ThreadDataLike,
    _ReadableThreadDataLike,
)


@_portable_group_operation(
    "shuffle",
    group_kinds=("block",),
)
def shuffle(
    group: ThreadGroup,
    value: _ReadableThreadDataLike[Any],
    /,
    *,
    mode: Any = "down",
    distance: Any = 1,
) -> ThreadDataLike[Any]:
    """Unit-shift a per-thread payload within a complete block."""

    raise CoopCompilerContextRequiredError(
        "cuda.coop.shuffle must be called from a supported GPU kernel."
    )


__all__ = ["shuffle"]
