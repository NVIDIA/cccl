# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose cooperative operations and activate their Numba-CUDA-MLIR planner.

Import this qualified API on the host before compiling kernels. Importing it
loads the supported compiler runtime and registers the whole-function
planner. Load, Store, and ThreadData are kernel constructs. Group
descriptors can also be created on the host and used as kernel globals.
Construct TempStorage inside each kernel. During compilation, the compiler
rebuilds the descriptor from its compile-time constant arguments and
validates them. The compiler rejects a descriptor that comes from a module
global. The ``local`` and ``shared`` array namespaces are resolved and
cached on first access.
"""

import importlib
from typing import TYPE_CHECKING

from .._core.api import TempStorageLike, ThreadDataLike
from ._compiler._activation import _initialize_runtime_hooks
from ._group._load_store import load, store
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

if TYPE_CHECKING:
    from ._thread_data import local, shared

__all__ = [
    "Hierarchy",
    "TempStorage",
    "TempStorageLike",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "exchange",
    "load",
    "local",
    "shared",
    "shuffle",
    "store",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]


def __getattr__(name):
    """Resolve and cache operation exports or array namespaces on first use."""

    if name in {"exchange", "shuffle"}:
        module_name = {
            "exchange": "_group_exchange",
            "shuffle": "_group_shuffle",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


_initialize_runtime_hooks()
