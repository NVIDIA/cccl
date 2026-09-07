# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR-qualified group-first cooperative primitives."""

from ._dataclass import gpu_dataclass, gpu_dataclass_argument_handler
from ._enums import (
    BlockLoadAlgorithm,
    BlockScanAlgorithm,
    BlockStoreAlgorithm,
    WarpLoadAlgorithm,
    WarpStoreAlgorithm,
)
from ._group_load_store import load, store
from ._group_reduce import reduce, sum
from ._group_scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from ._stateful_function import StatefulFunction
from ._temp_storage import TempStorage
from ._thread_data import ThreadData, local, shared
from ._thread_group import (
    Hierarchy,
    ThreadGroup,
    ThreadHierarchy,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)

__all__ = [
    "BlockLoadAlgorithm",
    "BlockScanAlgorithm",
    "BlockStoreAlgorithm",
    "Hierarchy",
    "StatefulFunction",
    "TempStorage",
    "ThreadData",
    "ThreadGroup",
    "ThreadHierarchy",
    "WarpLoadAlgorithm",
    "WarpStoreAlgorithm",
    "exclusive_scan",
    "exclusive_sum",
    "gpu_dataclass",
    "gpu_dataclass_argument_handler",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "local",
    "reduce",
    "scan",
    "shared",
    "store",
    "sum",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]
