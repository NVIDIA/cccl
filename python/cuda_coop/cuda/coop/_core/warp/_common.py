# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validation shared by backend-neutral warp primitive builders."""

from __future__ import annotations


def _validate_items_per_thread(items_per_thread: int) -> int:
    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread < 1
    ):
        raise ValueError("items_per_thread must be a positive integer")
    return items_per_thread


def _validate_logical_warp_threads(threads_in_warp: int) -> int:
    if (
        not isinstance(threads_in_warp, int)
        or isinstance(threads_in_warp, bool)
        or threads_in_warp < 1
        or threads_in_warp > 32
        or threads_in_warp & (threads_in_warp - 1)
    ):
        raise ValueError("threads_in_warp must be a power of two in [1, 32]")
    return threads_in_warp
