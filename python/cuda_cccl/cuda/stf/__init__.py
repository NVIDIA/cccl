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
        "jit",
        "has_numba",
        "has_numba_cuda",
        "has_torch",
    ]

    def __getattr__(name: str):
        """Lazy-load jit so numba-cuda is only required when using the decorator."""
        if name == "jit":
            try:
                from .decorator import jit as _jit

                return _jit
            except ImportError as e:
                raise AttributeError(
                    "cuda.stf.jit requires numba-cuda. Install with: pip install numba-cuda"
                ) from e
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def has_torch() -> bool:
        import importlib.util

        return importlib.util.find_spec("torch") is not None

    def has_numba() -> bool:
        import importlib.util

        return importlib.util.find_spec("numba") is not None

    def has_numba_cuda() -> bool:
        """True if numba-cuda is available (required for the jit decorator)."""
        try:
            from numba import cuda  # noqa: F401

            return True
        except ImportError:
            return False
