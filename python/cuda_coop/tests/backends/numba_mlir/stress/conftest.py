# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def require_numba_mlir_cuda_device():
    """Probe the GPU when stress tests start, not while pytest collects them."""

    cuda = pytest.importorskip("numba_cuda_mlir.cuda")
    if not cuda.is_available():
        pytest.skip("CUDA GPU required for Numba-CUDA-MLIR stress tests")
