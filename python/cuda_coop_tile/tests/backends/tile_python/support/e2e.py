# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run primitive tests with installed cuTile and the selected CUDA toolkit."""

import os
from pathlib import Path

import pytest


def configure_e2e_environment(monkeypatch):
    from cuda.coop.tile._cpp import _get_tile_library_linker_input

    # Missing libraries fail instead of silently dropping runtime coverage.
    _get_tile_library_linker_input()
    toolkit = os.environ.get("CUDA_COOP_TILE_CUDA_ROOT")
    if toolkit:
        root = Path(toolkit).resolve()
        if not (root / "bin/nvcc").is_file():
            pytest.fail(
                f"CUDA_COOP_TILE_CUDA_ROOT is not a CUDA toolkit: {root}"
            )
        for name in (
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDA_ROOT",
            "CUDA_TOOLKIT_PATH",
        ):
            monkeypatch.setenv(name, str(root))
        monkeypatch.setenv(
            "PATH", str(root / "bin") + os.pathsep + os.environ.get("PATH", "")
        )


def import_e2e_modules():
    import cuda.tile as ct
    import torch

    import cuda.coop.tile as coop

    if not torch.cuda.is_available():
        pytest.fail("Tile runtime tests require a CUDA GPU visible to torch")
    expected = os.environ.get("CUDA_COOP_TILE_E2E_SM_ARCH")
    if expected:
        major, minor = torch.cuda.get_device_capability()
        assert expected == f"sm_{major}{minor}", (
            "wrong GPU architecture selected"
        )
    return torch, ct, coop
