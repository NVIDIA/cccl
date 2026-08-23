# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS-backed group-first cooperative primitives."""

import importlib as _importlib

from cuda.coop._core.root_api import (
    _register_qualified_backend as _register_qualified_backend,
)

from ._compiler import register_trace_context as _register_trace_context
from ._group_load_store import load, store
from ._group_reduce import reduce, sum
from ._group_scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from ._internal import (
    ThreadData,
    ThreadDataLoadSource,
    ThreadDataSource,
    ThreadDataTensorMetadata,
)
from ._payload import Payload
from ._runtime_dependency import (
    validate_cutlass_runtime as _validate_cutlass_runtime,
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
    "Payload",
    "TempStorage",
    "ThreadData",
    "ThreadDataLoadSource",
    "ThreadDataSource",
    "ThreadDataTensorMetadata",
    "ThreadGroup",
    "ThreadHierarchy",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "reduce",
    "scan",
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
    if name in {"_block", "_warp"}:
        module = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


_register_qualified_backend(__name__)
