# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declare a fixed-size per-thread payload for the kernel compiler."""

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError
from ._payload import ThreadDataLike


def ThreadData(
    items_per_thread: int,
    dtype: object = None,
    *,
    alignment: int | None = None,
) -> ThreadDataLike[Any]:
    """Construct a per-thread payload with optional minimum storage alignment.

    ``alignment`` is a compile-time positive power of two in bytes, or ``None``
    to let the compiler choose. It applies when payload storage is materialized;
    it does not assert alignment of the inputs or outputs of Load and Store.
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.ThreadData must be called from a supported GPU kernel."
    )


__all__ = ["ThreadData", "ThreadDataLike"]
