# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for explicit CUTLASS temporary storage."""

from .._typing import TempStorageSharing

class TempStorage:
    """Explicit CUTLASS shared-memory scratch planner."""

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool
    sharing: TempStorageSharing

    def __init__(
        self,
        size_in_bytes: int | None = None,
        alignment: int | None = None,
        auto_sync: bool | None = None,
        sharing: TempStorageSharing = "shared",
    ) -> None:
        """Configure scratch capacity, alignment, synchronization, and sharing."""

    @property
    def required_size_in_bytes(self) -> int:
        """Return scratch bytes required by recorded collective uses."""

    @property
    def capacity_size_in_bytes(self) -> int | None:
        """Return the explicit or planned scratch capacity."""

    @property
    def required_alignment(self) -> int:
        """Return the strongest alignment required by recorded uses."""

    def sync(self) -> None:
        """Synchronize threads that may reuse this scratch allocation."""

__all__ = ["TempStorage"]
