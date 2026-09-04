# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose cooperative operations and activate their Numba-CUDA-MLIR planner.

Import this qualified API on the host before compiling kernels. Importing it
loads the supported compiler runtime and registers the whole-function
planner. Primitive calls and ThreadData are kernel constructs. Group
descriptors can also be created on the host and used as kernel globals.
Construct TempStorage inside each kernel. During compilation, the compiler
rebuilds the descriptor from its compile-time constant arguments and
validates them. The compiler rejects a descriptor that comes from a module
global. The ``local`` and ``shared`` namespaces and the Exchange, Shuffle,
Reduce, Sum, and Scan markers load on first access.
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
    from ._group._exchange import exchange
    from ._group._reduce import reduce, sum
    from ._group._scan import (
        exclusive_scan,
        exclusive_sum,
        inclusive_scan,
        inclusive_sum,
        scan,
    )
    from ._group._shuffle import shuffle
    from ._thread_data import local, shared

__all__ = [
    "Hierarchy",
    "StatefulFunction",
    "TempStorage",
    "TempStorageLike",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "local",
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
    """Load optional operation markers and allocation helpers on first use.

    Cache the resolved export in this module so later access reuses the same
    callable. This keeps their imports out of basic namespace initialization.
    Unknown names raise ``AttributeError`` as normal module lookup requires.
    """

    if name in {
        "exchange",
        "exclusive_scan",
        "exclusive_sum",
        "inclusive_scan",
        "inclusive_sum",
        "reduce",
        "scan",
        "shuffle",
        "sum",
    }:
        module_name = {
            "exchange": "_group._exchange",
            "exclusive_scan": "_group._scan",
            "exclusive_sum": "_group._scan",
            "inclusive_scan": "_group._scan",
            "inclusive_sum": "_group._scan",
            "reduce": "_group._reduce",
            "scan": "_group._scan",
            "shuffle": "_group._shuffle",
            "sum": "_group._reduce",
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
