# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose cooperative operations and activate their Numba-CUDA-MLIR planner.

Import this qualified API on the host before compiling kernels. Importing it
loads the supported compiler runtime and registers the compiler pass that
plans and replaces cooperative calls across each kernel before typing.
Primitive calls and ThreadData are kernel constructs. Group descriptors can
also be created on the host and used as kernel globals. StatefulFunction
descriptors are created on the host and used as kernel constants. Construct
TempStorage inside each kernel. During compilation, the compiler rebuilds
the descriptor from its compile-time constant arguments and validates them.
The compiler rejects a descriptor that comes from a module global. The
module loads Load and Store during initialization. Other operation markers,
the ``local`` and ``shared`` namespaces, and ``StatefulFunction`` load on
first access.
"""

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
    "Hierarchy",
    "StatefulFunction",
    "TempStorage",
    "TempStorageLike",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "adjacent_difference",
    "discontinuity",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
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
    "run_length_decode_into",
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


def __getattr__(name):
    """Load operation markers, allocation namespaces, and state descriptors.

    Resolve these exports on first use and cache each object in this module.
    This keeps their imports out of basic namespace initialization.
    Unknown names raise ``AttributeError`` as normal module lookup requires.
    """

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
        "run_length_decode",
        "run_length_decode_into",
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
            "run_length_decode": "_group_run_length",
            "run_length_decode_into": "_group_run_length",
            "scan": "_group_scan",
            "shuffle": "_group_shuffle",
            "sum": "_group_reduce",
            "topk_max_keys": "_group_topk",
            "topk_max_pairs": "_group_topk",
            "topk_min_keys": "_group_topk",
            "topk_min_pairs": "_group_topk",
        }[name]
        value = getattr(
            importlib.import_module(f"{__name__}.{module_name}"), name
        )
        globals()[name] = value
        return value
    if name in {"local", "shared"}:
        value = getattr(
            importlib.import_module(f"{__name__}._thread_data"), name
        )
        globals()[name] = value
        return value
    if name == "StatefulFunction":
        value = getattr(
            importlib.import_module(f"{__name__}._stateful_function"), name
        )
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


_initialize_runtime_hooks()
