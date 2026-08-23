# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Launch-qualified compilation helpers for Numba-CUDA-MLIR tests."""

from __future__ import annotations

import warnings
from typing import Any


def _dim3(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    return (value, 1, 1) if isinstance(value, int) else value


def compile_for_launch(
    dispatcher: Any,
    signature: Any,
    *,
    block: int | tuple[int, int, int],
    grid: int | tuple[int, int, int] = 1,
    sharedmem: int = 0,
    cluster: int | tuple[int, int, int] | None = None,
) -> tuple[Any, Any]:
    """Compile one launch-qualified specialization without launching it.

    Numba-CUDA-MLIR does not yet expose a public configured-compile method. Keep
    the private compiler dependency isolated here so evidence tests have one
    compatibility boundary to replace when the compiler grows that API.
    """

    from numba_cuda_mlir import descriptor
    from numba_cuda_mlir.numba_cuda.core.errors import NumbaPerformanceWarning

    config = {
        "grid": _dim3(grid),
        "block": _dim3(block),
        "sharedmem": sharedmem,
        "cluster": None if cluster is None else _dim3(cluster),
    }
    launch_key = descriptor._launch_config_key(config)
    compiler = getattr(dispatcher, "_compile_launch_config_signature", None)
    if not callable(compiler):
        raise RuntimeError(
            "Numba-CUDA-MLIR runtime lacks launch-qualified compile support"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Persistent disk cache is disabled for "
                "launch-config-specialized compiles"
            ),
            category=NumbaPerformanceWarning,
        )
        result = compiler(signature, launch_key)

    matching = tuple(
        key
        for key in dispatcher.signatures
        if getattr(key, "launch_config_key", None) == launch_key
    )
    if len(matching) != 1:
        raise AssertionError(
            f"expected one launch-qualified signature, found {len(matching)}"
        )
    return matching[0], result
