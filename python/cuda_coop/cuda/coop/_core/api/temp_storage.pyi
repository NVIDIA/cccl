# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable explicit temporary storage."""

from cuda.coop._typing import TempStorageLike, TempStorageSharing

__all__ = ["TempStorage", "TempStorageLike"]

def TempStorage(
    size_in_bytes: int | None = None,
    alignment: int | None = None,
    auto_sync: bool | None = None,
    sharing: TempStorageSharing = "shared",
) -> TempStorageLike:
    """Construct the selected backend's explicit scratch descriptor."""
