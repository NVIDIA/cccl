# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared helpers for the typed-pipeline tests.

The pipeline is chosen once per process, when ``cuda.coop.numba_mlir``
activates. Tests that compile kernels in process need typed mode and skip
otherwise; run them with ``CUDA_COOP_NUMBA_MLIR_PIPELINE=typed``. Tests of
activation itself start a fresh interpreter per case and run in any mode.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

TYPED = (
    os.environ.get("CUDA_COOP_NUMBA_MLIR_PIPELINE", "").strip().lower()
    == "typed"
)

requires_typed_pipeline = pytest.mark.skipif(
    not TYPED, reason="needs CUDA_COOP_NUMBA_MLIR_PIPELINE=typed"
)

# Compile-only tests pin the target and stand in for the launch's block.
CHIP = "sm_120"
BLOCK = (128, 1, 1)

ALGORITHMS = (
    "direct",
    "striped",
    "vectorize",
    "transpose",
    "warp_transpose",
    "warp_transpose_timesliced",
)
STORAGE_FREE = ALGORITHMS[:3]
TRANSPOSING = ALGORITHMS[3:]


def f32_in_out():
    """The signature ``(float32[::1], float32[::1])``."""

    from numba_cuda_mlir import types

    return types.void(types.float32[::1], types.float32[::1])


def run_python(code: str, pipeline: str | None) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter with the given pipeline variable."""

    env = dict(os.environ)
    env.pop("CUDA_COOP_NUMBA_MLIR_PIPELINE", None)
    if pipeline is not None:
        env["CUDA_COOP_NUMBA_MLIR_PIPELINE"] = pipeline
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
