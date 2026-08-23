# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable

import pytest


@pytest.fixture(scope="session")
def numba_mlir_cuda_available(
    backend_prerequisite: Callable[[str, bool, str], None],
) -> None:
    """Check Numba-CUDA-MLIR GPU availability during test setup."""

    cuda = pytest.importorskip("numba_cuda_mlir.cuda")
    backend_prerequisite(
        "numba_mlir",
        cuda.is_available(),
        "CUDA GPU required for Numba-CUDA-MLIR tests",
    )
