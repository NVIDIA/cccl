# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import block, warp
from ._array import local, shared
from ._dataclass import gpu_dataclass
from ._enums import (
    BlockHistogramAlgorithm,
    BlockLoadAlgorithm,
    BlockScanAlgorithm,
    BlockStoreAlgorithm,
    NoAlgorithm,
    WarpLoadAlgorithm,
    WarpStoreAlgorithm,
)
from ._numba_extension import _init_extension
from ._types import (
    Decomposer,
    StatefulFunction,
    TempStorage,
    ThreadData,
)

__all__ = [
    "_init_extension",
    "block",
    "BlockHistogramAlgorithm",
    "BlockLoadAlgorithm",
    "BlockScanAlgorithm",
    "BlockStoreAlgorithm",
    "gpu_dataclass",
    "local",
    "NoAlgorithm",
    "Decomposer",
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
