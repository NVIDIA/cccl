# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative exchange entry point."""

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
    "exchange",
    group_kinds=("block", "warp", "threads_within_warp"),
)
def exchange(
    group: ThreadGroup,
    value: _ReadableThreadDataLike[Any],
    /,
    *,
    mode: Any = "striped_to_blocked",
) -> ThreadDataLike[Any]:
    """Rearrange a per-thread payload within the selected group."""

    raise CoopCompilerContextRequiredError(
        "cuda.coop.exchange must be called from a supported GPU kernel."
    )


__all__ = ["exchange"]
