# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from cuda.cccl.cooperative.experimental import block, warp
from cuda.cccl.cooperative.experimental._array import local, shared
from cuda.cccl.cooperative.experimental._enums import (
    BlockLoadAlgorithm,
    BlockStoreAlgorithm,
    WarpLoadAlgorithm,
    WarpStoreAlgorithm,
)
from cuda.cccl.cooperative.experimental._numba_extension import _init_extension
from cuda.cccl.cooperative.experimental._types import StatefulFunction

__all__ = [
    "_init_extension",
    "block",
    "BlockLoadAlgorithm",
    "BlockStoreAlgorithm",
    "local",
    "shared",
    "StatefulFunction",
    "warp",
    "WarpLoadAlgorithm",
    "WarpStoreAlgorithm",
]

# Our extension initialization doesn't appear to be getting called
# automatically, despite the pyproject.toml entry, so, manually
# call it here.
_init_extension()
