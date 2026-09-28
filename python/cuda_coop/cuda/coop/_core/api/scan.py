# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative scan entry points."""

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup
from ._dispatch import (
    _portable_group_operation,
)
from ._payload import (
    TempStorageLike,
)

_PORTABLE_SCAN_GROUP_KINDS = ("block", "warp", "threads_within_warp")


@_portable_group_operation("scan", group_kinds=_PORTABLE_SCAN_GROUP_KINDS)
def scan(
    group: ThreadGroup,
    value: object,
    /,
    *,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: str | None = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Scan values across a block or warp group through the active backend."""

    raise CoopCompilerContextRequiredError(
        "cuda.coop.scan must be called from a supported GPU kernel."
    )


@_portable_group_operation(
    "exclusive_sum",
    group_kinds=_PORTABLE_SCAN_GROUP_KINDS,
)
def exclusive_sum(
    group: ThreadGroup,
    value: object,
    /,
    *,
    algorithm: str | None = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Return an out-of-place exclusive prefix sum."""

    raise CoopCompilerContextRequiredError(
        "cuda.coop.exclusive_sum must be called from a supported GPU kernel."
    )


@_portable_group_operation(
    "inclusive_sum",
    group_kinds=_PORTABLE_SCAN_GROUP_KINDS,
)
def inclusive_sum(
    group: ThreadGroup,
    value: object,
    /,
    *,
    algorithm: str | None = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Return an out-of-place inclusive prefix sum."""

    raise CoopCompilerContextRequiredError(
        "cuda.coop.inclusive_sum must be called from a supported GPU kernel."
    )


@_portable_group_operation(
    "exclusive_scan",
    group_kinds=_PORTABLE_SCAN_GROUP_KINDS,
)
def exclusive_scan(
    group: ThreadGroup,
    value: object,
    /,
    *,
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: str | None = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Return an out-of-place exclusive scan."""

    raise CoopCompilerContextRequiredError(
        "cuda.coop.exclusive_scan must be called from a supported GPU kernel."
    )


@_portable_group_operation(
    "inclusive_scan",
    group_kinds=_PORTABLE_SCAN_GROUP_KINDS,
)
def inclusive_scan(
    group: ThreadGroup,
    value: object,
    /,
    *,
    scan_op: Any = None,
    algorithm: str | None = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Return an out-of-place inclusive scan."""

    raise CoopCompilerContextRequiredError(
        "cuda.coop.inclusive_scan must be called from a supported GPU kernel."
    )


__all__ = [
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "scan",
]
