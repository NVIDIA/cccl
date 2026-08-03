# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUDA Core Compute Libraries header package."""

from .headers import IncludePaths, __version__, get_include_paths

__all__ = ["IncludePaths", "__version__", "get_include_paths"]
