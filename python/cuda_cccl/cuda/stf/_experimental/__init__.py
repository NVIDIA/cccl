# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Experimental Python bindings for CUDASTF (Stream Task Flow)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from . import paths
from .paths import get_include_paths, get_library_dir, get_library_path

# Map each lazily-exported public symbol to the submodule that defines it.
# Importing those submodules pulls in the STF extension (_stf_bindings_impl)
# and preloads CUDA libraries, so we defer it until a symbol is first accessed.
# This keeps `import cuda.stf._experimental.paths` (path discovery) cheap.
_LAZY_SYMBOLS = {
    "AccessMode": "._stf_bindings",
    "CudaStream": "._stf_bindings",
    "async_resources": "._stf_bindings",
    "context": "._stf_bindings",
    "data_place": "._stf_bindings",
    "dep": "._stf_bindings",
    "exec_place": "._stf_bindings",
    "exec_place_grid": "._stf_bindings",
    "exec_place_resources": "._stf_bindings",
    "green_context_helper": "._stf_bindings",
    "green_ctx_view": "._stf_bindings",
    "machine_init": "._stf_bindings",
    "stackable_context": "._stf_bindings",
    "DeviceArray": ".device_array",
    "TaskGraph": ".task_graph",
    "task_graph": ".task_graph",
}

if TYPE_CHECKING:
    from ._stf_bindings import (
        AccessMode,
        CudaStream,
        async_resources,
        context,
        data_place,
        dep,
        exec_place,
        exec_place_grid,
        exec_place_resources,
        green_context_helper,
        green_ctx_view,
        machine_init,
        stackable_context,
    )
    from .device_array import DeviceArray
    from .task_graph import TaskGraph, task_graph


def __getattr__(name: str) -> Any:
    module_name = _LAZY_SYMBOLS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    # Cache on the module so subsequent lookups skip __getattr__.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AccessMode",
    "CudaStream",
    "DeviceArray",
    "TaskGraph",
    "async_resources",
    "context",
    "dep",
    "exec_place",
    "exec_place_grid",
    "exec_place_resources",
    "get_include_paths",
    "get_library_dir",
    "get_library_path",
    "green_context_helper",
    "green_ctx_view",
    "data_place",
    "machine_init",
    "paths",
    "stackable_context",
    "task_graph",
]
