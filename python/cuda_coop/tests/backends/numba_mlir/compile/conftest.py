# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Select mode-specific compiler expectations without changing activation."""

import os

import pytest


@pytest.fixture(scope="session")
def typed_pipeline():
    return (
        os.environ.get("CUDA_COOP_NUMBA_MLIR_PIPELINE", "").strip().lower()
        == "typed"
    )
