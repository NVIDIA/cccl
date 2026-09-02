# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral physical-warp cooperative primitive descriptions."""

from .reduce import WarpReduceOperation, WarpReduceSpec, make_warp_reduce_spec

__all__ = [
    "WarpReduceOperation",
    "WarpReduceSpec",
    "make_warp_reduce_spec",
]
