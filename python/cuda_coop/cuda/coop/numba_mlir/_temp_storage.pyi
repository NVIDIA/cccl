# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any

from .._typing import TempStorageSharing

class TempStorage:
    """Explicit opaque byte scratch for planned shared-memory operations."""

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

    def release(self, dtype: object | None = None) -> Any:
        """Consume the descriptor and return its complete shared array."""

__all__ = ["TempStorage"]
