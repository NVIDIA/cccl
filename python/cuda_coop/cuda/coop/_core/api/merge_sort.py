# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Merge Sort entry points."""

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup
from ._dispatch import (
    _portable_group_operation,
)
from ._payload import (
    TempStorageLike,
    ThreadDataLike,
    _ReadableThreadDataLike,
)


@_portable_group_operation(
    "merge_sort_keys", group_kinds=("block", "warp", "threads_within_warp")
)
def merge_sort_keys(
    group: ThreadGroup,
    keys: _ReadableThreadDataLike[Any],
    /,
    *,
    descending: bool = False,
    valid_items: object = None,
    oob_default: object = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[Any]:
    """Return sorted blocked payloads without modifying the inputs."""

    raise CoopCompilerContextRequiredError(
        "cuda.coop.merge_sort_keys must be called from a supported GPU kernel."
    )


@_portable_group_operation(
    "merge_sort_pairs", group_kinds=("block", "warp", "threads_within_warp")
)
def merge_sort_pairs(
    group: ThreadGroup,
    keys: _ReadableThreadDataLike[Any],
    values: _ReadableThreadDataLike[Any],
    /,
    *,
    descending: bool = False,
    valid_items: object = None,
    oob_default: object = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadDataLike[Any], ThreadDataLike[Any]]:
    """Return sorted blocked payloads without modifying the inputs."""

    raise CoopCompilerContextRequiredError(
        "cuda.coop.merge_sort_pairs must be called from a supported GPU kernel."
    )


__all__ = ["merge_sort_keys", "merge_sort_pairs"]
