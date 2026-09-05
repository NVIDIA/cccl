# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def require_numba_mlir_cuda_device(numba_mlir_cuda_available):
    """Require the backend fixture when runtime tests start."""
