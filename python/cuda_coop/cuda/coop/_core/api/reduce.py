# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative reduction entry points."""

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup
from ._dispatch import (
    _portable_group_operation,
)

_PORTABLE_REDUCTION_GROUP_KINDS = (
    "thread",
    "warp",
    "threads_within_warp",
    "block",
    "warps_within_block",
    "cluster",
)


@_portable_group_operation(
    "reduce",
    group_kinds=_PORTABLE_REDUCTION_GROUP_KINDS,
)
def reduce(
    group: ThreadGroup,
    value: object,
    /,
    *,
    binary_op: Any = None,
    broadcast: bool = True,
    valid_items: object = None,
    algorithm: str | None = None,
) -> Any:
    """Reduce values across a group through the compiler-selected backend.

    With ``broadcast=False``, only group rank zero has a defined result. Every
    member must still participate in the collective.
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.reduce must be called from a supported GPU kernel."
    )


@_portable_group_operation(
    "sum",
    group_kinds=_PORTABLE_REDUCTION_GROUP_KINDS,
)
def sum(
    group: ThreadGroup,
    value: object,
    /,
    *,
    broadcast: bool = True,
    valid_items: object = None,
    algorithm: str | None = None,
) -> Any:
    """Sum values across a group through the compiler-selected backend.

    With ``broadcast=False``, only group rank zero has a defined result. Every
    member must still participate in the collective.
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.sum must be called from a supported GPU kernel."
    )


__all__ = ["reduce", "sum"]
