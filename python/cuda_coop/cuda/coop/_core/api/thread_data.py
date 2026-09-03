# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable per-thread payload construction.

ThreadData is a compiler-owned fixed-size value container; this frontend only
forwards its static extent and optional dtype to the active backend. Primitive
payload validation lives in the family frontends and shared payload helpers.
"""

from __future__ import annotations

from typing import Any

from ._dispatch import _backend_member, _common_root_operation_scope
from ._payload import ThreadDataLike


def ThreadData(items_per_thread: int, dtype: Any = None) -> ThreadDataLike[Any]:
    """Construct the selected backend's per-thread payload container."""

    with _common_root_operation_scope("ThreadData"):
        return _backend_member("ThreadData")(items_per_thread, dtype=dtype)


__all__ = ["ThreadData", "ThreadDataLike"]
