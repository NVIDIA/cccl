# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import block, warp
from ._array import local, shared
from ._enums import (
    BlockLoadAlgorithm,
    BlockStoreAlgorithm,
    WarpLoadAlgorithm,
    WarpStoreAlgorithm,
)
from ._numba_extension import _init_extension
from ._types import (
    StatefulFunction,
    TempStorage,
    ThreadData,
)

__all__ = [
    "_init_extension",
    "block",
    "BlockLoadAlgorithm",
    "BlockStoreAlgorithm",
    "local",
    "shared",
    "StatefulFunction",
    "TempStorage",
    "ThreadData",
    "warp",
    "WarpLoadAlgorithm",
    "WarpStoreAlgorithm",
]

# Our extension initialization doesn't appear to be getting called
# automatically, despite the pyproject.toml entry, so, manually
# call it here.
_init_extension()
