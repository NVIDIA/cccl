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
    StatefulFunction,
    TempStorage,
    ThreadData,
)
from .block._block_load_store import (
    BlockLoad,
    BlockStore,
)

__all__ = [
    "_init_extension",
    "block",
    "BlockHistogramAlgorithm",
    "BlockLoad",
    "BlockLoadAlgorithm",
    "BlockScanAlgorithm",
    "BlockStoreAlgorithm",
    "BlockStore",
    "gpu_dataclass",
    "local",
    "NoAlgorithm",
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
