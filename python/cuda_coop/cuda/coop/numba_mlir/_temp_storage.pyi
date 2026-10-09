# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any

from .._typing import TempStorageSharing

class TempStorage:
    """Explicit scratch and typed reservations planned in shared memory."""

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool
    sharing: TempStorageSharing

    def __init__(
        self,
        size_in_bytes: int | None = None,
        *,
        alignment: int | None = None,
        auto_sync: bool | None = False,
        sharing: TempStorageSharing = "shared",
    ) -> None:
        """Configure scratch size, alignment, synchronization, and sharing."""

    def reserve(
        self, num_elems: int, dtype: object, *, alignment: int | None = None
    ) -> Any:
        """Return a compiler shared array; requires manual synchronization."""

__all__ = ["TempStorage"]
