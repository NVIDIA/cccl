# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any

from .._typing import TempStorageSharing

class TempStorage:
    """Shared-memory requirements for primitive scratch and typed arrays."""

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

    @property
    def capacity_size_in_bytes(self) -> int | None: ...
    @property
    def is_deferred(self) -> bool: ...
    def sync(self) -> None:
        """Synchronize the block before manually reusing temporary storage."""
    def reserve(
        self,
        num_elems: int,
        dtype: object,
        *,
        alignment: int | None = None,
    ) -> Any:
        """Return a typed CuTe tensor using this descriptor's sharing policy."""

__all__ = ["TempStorage"]
