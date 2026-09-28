# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declare shared-memory requirements for the kernel compiler."""

from __future__ import annotations

from ..thread_group import CoopCompilerContextRequiredError
from ._payload import TempStorageLike


def TempStorage(
    size_in_bytes: int | None = None,
    *,
    alignment: int | None = None,
    auto_sync: bool | None = None,
    sharing: str = "shared",
) -> TempStorageLike:
    """Construct scratch storage with optional minimum alignment in bytes.

    ``alignment`` is a compile-time positive power of two, or ``None`` to let
    the compiler choose. Storage satisfies both this minimum and the alignment
    required by every primitive using it.
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.TempStorage must be called from a supported GPU kernel."
    )


__all__ = ["TempStorage", "TempStorageLike"]
