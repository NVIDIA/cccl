# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from cuda.cccl._version import __version__
from cuda.cccl.include_paths import get_include_paths

__all__ = ["__version__", "get_include_paths"]
