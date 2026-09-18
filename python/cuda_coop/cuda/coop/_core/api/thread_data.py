# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable per-thread payload construction.

ThreadData is a compiler-owned fixed-size value container; this frontend only
forwards its static extent, optional dtype, and alignment to the active backend.
Primitive payload validation lives in the family frontends and shared helpers.
"""

from __future__ import annotations

from typing import Any

from ._dispatch import _backend_member, _common_root_operation_scope
from ._payload import ThreadDataLike, _normalize_alignment


def ThreadData(
    items_per_thread: int,
    dtype: Any = None,
    *,
    alignment: int | None = None,
) -> ThreadDataLike[Any]:
    """Construct a per-thread payload with optional minimum storage alignment.

    ``alignment`` is a compile-time positive power of two in bytes, or ``None``
    to let the compiler choose. It applies when payload storage is materialized;
    it does not assert alignment of the inputs or outputs of Load and Store.
    """

    alignment = _normalize_alignment(alignment)

    with _common_root_operation_scope("ThreadData"):
        return _backend_member("ThreadData")(
            items_per_thread, dtype=dtype, alignment=alignment
        )


__all__ = ["ThreadData", "ThreadDataLike"]
