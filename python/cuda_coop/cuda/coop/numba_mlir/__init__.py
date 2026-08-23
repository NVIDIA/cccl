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

__all__ = [
    "Hierarchy",
    "StatefulFunction",
    "TempStorage",
    "ThreadData",
    "ThreadGroup",
    "ThreadHierarchy",
    "gpu_dataclass",
    "gpu_dataclass_argument_handler",
    "local",
    "shared",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


_initialize_runtime_hooks()
