# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Programmatic access to the CCCL headers bundled in this wheel."""

import importlib.metadata

from .include_paths import IncludePaths, get_include_paths

try:
    __version__ = importlib.metadata.version("cccl-headers")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["IncludePaths", "__version__", "get_include_paths"]
