# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical symbol fragments shared by CUTLASS provider artifacts."""

from __future__ import annotations


def block_dim_token(block_dim: tuple[int, int, int]) -> str:
    """Return the existing provider ABI token for an exact CUDA block shape."""

    x, y, z = block_dim
    if (y, z) == (1, 1):
        return f"b{x}"
    return f"b{x}x{y}x{z}"


__all__ = ["block_dim_token"]
