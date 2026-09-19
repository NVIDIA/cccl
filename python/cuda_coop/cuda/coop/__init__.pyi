# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common cooperative primitives shared by supported CUDA Python DSLs."""

from typing import Literal

from ._core.api.exchange import exchange
from ._core.api.histogram import histogram as histogram
from ._core.api.load_store import load, store
from ._core.api.merge_sort import merge_sort_keys as merge_sort_keys
from ._core.api.merge_sort import merge_sort_pairs as merge_sort_pairs
from ._core.api.neighbors import adjacent_difference as adjacent_difference
from ._core.api.neighbors import discontinuity as discontinuity
from ._core.api.radix import radix_rank, radix_sort_keys, radix_sort_pairs
from ._core.api.reduce import reduce, sum
from ._core.api.reduce_batched import reduce_batched
from ._core.api.run_length import run_length_decode as run_length_decode
from ._core.api.run_length import run_length_decode_into as run_length_decode_into
from ._core.api.scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from ._core.api.shuffle import shuffle
from ._core.api.temp_storage import TempStorage, TempStorageLike
from ._core.api.thread_data import ThreadData, ThreadDataLike
from ._core.api.thread_group import (
    Hierarchy,
    ThreadGroup,
    ThreadHierarchy,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)
from ._core.api.topk import topk_max_keys, topk_max_pairs, topk_min_keys, topk_min_pairs

__version__: str

def register(backend: Literal["numba-cuda-mlir", "numba_cuda_mlir"]) -> None: ...

__all__ = [
    "topk_min_keys",
    "topk_min_pairs",
    "topk_max_keys",
    "topk_max_pairs",
    "Hierarchy",
    "TempStorage",
    "TempStorageLike",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "__version__",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_pairs",
    "reduce",
    "reduce_batched",
    "run_length_decode",
    "run_length_decode_into",
    "register",
    "scan",
    "shuffle",
    "store",
    "sum",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]
