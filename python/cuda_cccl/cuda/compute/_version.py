# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Distribution version for :mod:`cuda.compute`."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cuda-compute")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
