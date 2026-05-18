# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from ._stf_bindings import _BINDINGS_AVAILABLE  # type: ignore[attr-defined]

if not _BINDINGS_AVAILABLE:
    __all__ = ["_BINDINGS_AVAILABLE"]

    def __getattr__(name: str):
        raise AttributeError(
            f"Cannot access 'cuda.stf.{name}' because CUDASTF bindings are not available. "
            "This typically means you're running on a CPU-only machine without CUDA drivers installed, "
            "or that cuda-cccl was not built with STF support."
        )
else:
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

    __all__ = [
        "_BINDINGS_AVAILABLE",
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
        "green_context_helper",
        "green_ctx_view",
        "data_place",
        "machine_init",
        "stackable_context",
        "task_graph",
    ]
