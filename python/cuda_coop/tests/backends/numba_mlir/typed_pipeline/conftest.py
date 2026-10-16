# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Supply target metadata for compiler tests with all devices hidden."""

import os
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def fixed_target_without_devices(monkeypatch):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        return

    from numba_cuda_mlir import cuda, tools

    # NCM 0.5.4 queries the host capability even with an explicit chip.
    monkeypatch.setattr(
        tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (12, 0) if as_type is tuple else "sm_120",
    )
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(12, 0)),
    )
