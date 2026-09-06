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
    "load",
    "reduce",
    "shuffle",
    "store",
    "sum",
    "local",
    "shared",
]


def __getattr__(name):
    if name in {"exchange", "reduce", "shuffle", "sum"}:
        module_name = {
            "exchange": "_group_exchange",
            "reduce": "_group_reduce",
            "shuffle": "_group_shuffle",
            "sum": "_group_reduce",
        }[name]
        value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
        globals()[name] = value
        return value
    if name in {"local", "shared"}:
        value = getattr(importlib.import_module(f"{__name__}._thread_data"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


_initialize_runtime_hooks()
