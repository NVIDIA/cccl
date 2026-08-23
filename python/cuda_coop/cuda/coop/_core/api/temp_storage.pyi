# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable explicit temporary storage."""

from cuda.coop._typing import TempStorageLike as TempStorageLike
from cuda.coop._typing import TempStorageSharing as _TempStorageSharing

def TempStorage(
    size_in_bytes: int | None = None,
    alignment: int | None = None,
    auto_sync: bool | None = None,
    sharing: _TempStorageSharing = "shared",
) -> TempStorageLike:
    """Construct the selected backend's explicit scratch descriptor."""
