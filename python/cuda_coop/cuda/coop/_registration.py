# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Explicit registration of cooperative primitives with a compiler backend."""

from __future__ import annotations

import importlib
from typing import Literal


def register(backend: Literal["numba-cuda-mlir", "numba_cuda_mlir", "cutlass"]) -> None:
    """Register cooperative primitives with the selected compiler backend.

    Call this on the host before compiling a kernel, including when
    :mod:`cuda.coop` was imported before the compiler. Repeated calls are safe.
    Explicit registration also works when automatic registration is disabled
    with ``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION=1``.

    Parameters
    ----------
    backend
        Compiler backend to register. ``"numba_cuda_mlir"`` is an alias for
        ``"numba-cuda-mlir"``. Use ``"cutlass"`` for CuTe DSL kernels.

    Raises
    ------
    ValueError
        The backend name is unsupported.
    ImportError
        The backend adapter or its required compiler is unavailable or
        incompatible. Other backend initialization errors propagate unchanged.
    """

    if backend in ("numba-cuda-mlir", "numba_cuda_mlir"):
        module_name, label = "cuda.coop.numba_mlir", "Numba-CUDA-MLIR"
    elif backend == "cutlass":
        module_name, label = "cuda.coop.cutlass", "CUTLASS"
    else:
        raise ValueError(
            f"Unsupported cuda.coop backend {backend!r}; "
            "expected 'numba-cuda-mlir', 'numba_cuda_mlir', or 'cutlass'."
        )

    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        raise ImportError(
            f"This cuda-coop installation does not include the {label} "
            f"adapter. Install a cuda-coop version with {label} support."
        ) from error
