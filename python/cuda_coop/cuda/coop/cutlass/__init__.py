# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS-backed group-first cooperative primitives."""

import importlib as _importlib

from cuda.coop._core.api import (
    _register_qualified_backend as _register_qualified_backend,
)

from ._compiler import register_trace_context as _register_trace_context
from ._compiler._runtime import (
    validate_cutlass_runtime as _validate_cutlass_runtime,
)
from ._group_exchange import exchange
from ._group_load_store import load, store
from ._group_reduce import reduce, sum
from ._group_scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from ._group_shuffle import shuffle
from ._thread_data import (
    ThreadData,
    ThreadDataLoadSource,
    ThreadDataSource,
    ThreadDataTensorMetadata,
)
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

_validate_cutlass_runtime()
_register_trace_context()

__all__ = [
    "Hierarchy",
    "TempStorage",
    "ThreadData",
    "ThreadDataLoadSource",
    "ThreadDataSource",
    "ThreadDataTensorMetadata",
    "ThreadGroup",
    "ThreadHierarchy",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "reduce",
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


def __getattr__(name: str):
    if name == "TempStorage":
        module = _importlib.import_module(f"{__name__}._temp_storage")
        value = module.TempStorage
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


_register_qualified_backend(__name__)
