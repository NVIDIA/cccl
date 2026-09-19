# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR-qualified cooperative group building blocks."""

import importlib

from .._core.api import TempStorageLike, ThreadDataLike
from ._compiler._activation import _initialize_runtime_hooks
from ._group_load_store import load, store
from ._temp_storage import TempStorage
from ._thread_data import ThreadData
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
    "adjacent_difference",
    "discontinuity",
    "Hierarchy",
    "TempStorage",
    "TempStorageLike",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "histogram",
    "load",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_pairs",
    "reduce",
    "scan",
    "shuffle",
    "store",
    "sum",
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
    "StatefulFunction",
    "local",
    "shared",
]


def __getattr__(name):
    if name in {
        "adjacent_difference",
        "discontinuity",
        "histogram",
        "merge_sort_keys",
        "merge_sort_pairs",
        "radix_rank",
        "radix_sort_keys",
        "radix_sort_pairs",
        "exchange",
        "exclusive_scan",
        "exclusive_sum",
        "inclusive_scan",
        "inclusive_sum",
        "reduce",
        "scan",
        "shuffle",
        "sum",
        "topk_max_keys",
        "topk_max_pairs",
        "topk_min_keys",
        "topk_min_pairs",
    }:
        module_name = {
            "adjacent_difference": "_group_neighbors",
            "discontinuity": "_group_neighbors",
            "histogram": "_group_histogram",
            "merge_sort_keys": "_group_merge_sort",
            "merge_sort_pairs": "_group_merge_sort",
            "radix_rank": "_group_radix",
            "radix_sort_keys": "_group_radix",
            "radix_sort_pairs": "_group_radix",
            "exchange": "_group_exchange",
            "exclusive_scan": "_group_scan",
            "exclusive_sum": "_group_scan",
            "inclusive_scan": "_group_scan",
            "inclusive_sum": "_group_scan",
            "reduce": "_group_reduce",
            "scan": "_group_scan",
            "shuffle": "_group_shuffle",
            "sum": "_group_reduce",
            "topk_max_keys": "_group_topk",
            "topk_max_pairs": "_group_topk",
            "topk_min_keys": "_group_topk",
            "topk_min_pairs": "_group_topk",
        }[name]
        value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
        globals()[name] = value
        return value
    if name in {"local", "shared"}:
        value = getattr(importlib.import_module(f"{__name__}._thread_data"), name)
        globals()[name] = value
        return value
    if name == "StatefulFunction":
        value = getattr(importlib.import_module(f"{__name__}._stateful_function"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


_initialize_runtime_hooks()
