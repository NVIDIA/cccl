# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR-qualified group-first cooperative primitives."""

from ._dataclass import gpu_dataclass, gpu_dataclass_argument_handler
from ._enums import (
    BlockHistogramAlgorithm,
    BlockLoadAlgorithm,
    BlockScanAlgorithm,
    BlockStoreAlgorithm,
    WarpLoadAlgorithm,
    WarpStoreAlgorithm,
)
from ._group_adjacent_difference import adjacent_difference
from ._group_discontinuity import discontinuity
from ._group_exchange import exchange
from ._group_histogram import histogram
from ._group_load_store import load, store
from ._group_merge_sort import merge_sort_keys, merge_sort_pairs
from ._group_radix import radix_rank, radix_sort_keys, radix_sort_pairs
from ._group_reduce import reduce, sum
from ._group_run_length_decode import run_length_decode
from ._group_scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from ._group_shuffle import shuffle
from ._group_topk import topk_max_keys, topk_max_pairs, topk_min_keys, topk_min_pairs
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
    "BlockHistogramAlgorithm",
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
    "adjacent_difference",
    "discontinuity",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "gpu_dataclass",
    "gpu_dataclass_argument_handler",
    "histogram",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "local",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_pairs",
    "reduce",
    "run_length_decode",
    "scan",
    "shared",
    "shuffle",
    "store",
    "sum",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]
