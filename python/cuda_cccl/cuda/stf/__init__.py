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
        context,
        data_place,
        dep,
        exec_place,
    )

    __all__ = [
        "_BINDINGS_AVAILABLE",
        "context",
        "dep",
        "exec_place",
        "data_place",
    ]
