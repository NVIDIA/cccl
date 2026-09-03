# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable explicit temporary-storage construction.

This frontend delegates caller-selected size, alignment, synchronization, and
sharing controls to the active backend. Allocation layout and reuse barriers
remain backend compiler responsibilities.
"""

from __future__ import annotations

from typing import Any

from ._dispatch import _backend_member
from ._payload import TempStorageLike


def TempStorage(
    size_in_bytes: Any = None,
    alignment: Any = None,
    auto_sync: Any = None,
    sharing: str = "shared",
) -> TempStorageLike:
    """Construct the selected backend's explicit scratch override."""

    return _backend_member("TempStorage")(
        size_in_bytes=size_in_bytes,
        alignment=alignment,
        auto_sync=auto_sync,
        sharing=sharing,
    )


__all__ = ["TempStorage", "TempStorageLike"]
