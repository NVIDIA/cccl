# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR-qualified cooperative group building blocks."""

import importlib

from ._compiler._activation import _initialize_runtime_hooks
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

_GROUP_OPERATION_MODULES = {
    "exchange": "_group_exchange",
    "exclusive_scan": "_group_scan",
    "exclusive_sum": "_group_scan",
    "inclusive_scan": "_group_scan",
    "inclusive_sum": "_group_scan",
    "load": "_group_load_store",
    "merge_sort_keys": "_group_merge_sort",
    "merge_sort_pairs": "_group_merge_sort",
    "radix_rank": "_group_radix",
    "radix_sort_keys": "_group_radix",
    "radix_sort_pairs": "_group_radix",
    "reduce": "_group_reduce",
    "scan": "_group_scan",
    "shuffle": "_group_shuffle",
    "store": "_group_load_store",
    "sum": "_group_reduce",
}

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
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "gpu_dataclass",
    "gpu_dataclass_argument_handler",
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
]


def __getattr__(name):
    if name in {"local", "shared"}:
        value = getattr(importlib.import_module(f"{__name__}._thread_data"), name)
        globals()[name] = value
        return value
    if name == "StatefulFunction":
        value = importlib.import_module(
            f"{__name__}._stateful_function"
        ).StatefulFunction
        globals()[name] = value
        return value
    if name in {"gpu_dataclass", "gpu_dataclass_argument_handler"}:
        value = getattr(importlib.import_module(f"{__name__}._dataclass"), name)
        globals()[name] = value
        return value
    if name in _GROUP_OPERATION_MODULES:
        module_name = _GROUP_OPERATION_MODULES[name]
        value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
        globals()[name] = value
        return value
    if name in {
        "BlockLoadAlgorithm",
        "BlockScanAlgorithm",
        "BlockStoreAlgorithm",
        "WarpLoadAlgorithm",
        "WarpStoreAlgorithm",
    }:
        value = getattr(importlib.import_module(f"{__name__}._enums"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


_initialize_runtime_hooks()
