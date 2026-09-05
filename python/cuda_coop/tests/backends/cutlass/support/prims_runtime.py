# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ....support.paths import PACKAGE_ROOT
from ....support.toolchains.cutlass import (
    find_ptxas_with_ptx_93,
    ptxas_probe_skip_reason,
)

SOURCE_ROOT = PACKAGE_ROOT
CUTLASS_API_MODULE = "cuda.coop.cutlass"


def _prims_runtime_env(tmp_path: Path) -> dict[str, str]:
    pytest.importorskip("cutlass")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable PyTorch runtime")
    ptxas, _ = find_ptxas_with_ptx_93()
    if ptxas is None:
        pytest.skip(ptxas_probe_skip_reason())

    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SOURCE_ROOT}{os.pathsep}{python_path}" if python_path else str(SOURCE_ROOT)
    )
    env["PATH"] = f"{ptxas.parent}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT", "ltoir")
    env.setdefault(
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR",
        str(tmp_path / "cuda-coop-cutlass-prims-runtime-cache"),
    )
    return env


def _run_prims_example(
    example_name: str,
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SOURCE_ROOT / "examples" / "cutlass" / example_name),
        ],
        check=False,
        capture_output=True,
        env=_prims_runtime_env(tmp_path),
        text=True,
    )

    return result
