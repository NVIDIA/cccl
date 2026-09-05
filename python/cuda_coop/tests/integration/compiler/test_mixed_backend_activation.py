# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compilation isolation for a module containing both certified backends."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ...support.paths import PACKAGE_ROOT, TESTS_ROOT

pytestmark = [
    pytest.mark.backend_cutlass,
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
]

_MIXED_MODULE = TESTS_ROOT / "support" / "fixtures" / "mixed_backend_kernels.py"


def _require_backends() -> None:
    pytest.importorskip("cutlass.cute")
    torch = pytest.importorskip("torch")
    numba_cuda = pytest.importorskip("numba_cuda_mlir.cuda")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable PyTorch runtime")
    if not numba_cuda.is_available():
        pytest.skip("requires a CUDA-capable Numba-CUDA-MLIR runtime")


def _without_source_package_root(python_path: str | None) -> str | None:
    """Retain compiler overlays while excluding the cuda-coop source tree."""

    if not python_path:
        return None
    package_root = PACKAGE_ROOT.resolve()
    retained = []
    for entry in python_path.split(os.pathsep):
        try:
            resolved = Path(entry or ".").resolve()
        except OSError:
            retained.append(entry)
            continue
        if resolved != package_root:
            retained.append(entry)
    return os.pathsep.join(retained) or None


def test_installed_python_path_filter_preserves_compiler_overlays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compiler_overlay = tmp_path / "compiler-overlay"
    compiler_overlay.mkdir()
    monkeypatch.chdir(tmp_path)

    python_path = os.pathsep.join(
        (os.fspath(PACKAGE_ROOT), "", os.fspath(compiler_overlay))
    )
    assert _without_source_package_root(python_path) == os.pathsep.join(
        ("", os.fspath(compiler_overlay))
    )


@pytest.mark.parametrize("mode", ["cutlass-first", "numba-first", "concurrent"])
def test_mixed_backend_compilation_is_order_independent(
    mode: str,
    tmp_path: Path,
) -> None:
    _require_backends()
    env = os.environ.copy()
    if env.get("CUDA_COOP_EXAMPLES_USE_INSTALLED_CUDA_COOP") == "1":
        python_path = _without_source_package_root(env.get("PYTHONPATH"))
        if python_path is None:
            env.pop("PYTHONPATH", None)
        else:
            env["PYTHONPATH"] = python_path
    else:
        python_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{PACKAGE_ROOT}{os.pathsep}{python_path}"
            if python_path
            else str(PACKAGE_ROOT)
        )
    env["CUDA_CACHE_PATH"] = os.fspath(tmp_path / "cuda-cache")
    env["NUMBA_CACHE_DIR"] = os.fspath(tmp_path / "numba-cache")
    env["CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR"] = os.fspath(
        tmp_path / "cutlass-provider-cache"
    )
    env["CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT"] = "ltoir"

    result = subprocess.run(
        [sys.executable, os.fspath(_MIXED_MODULE), mode],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"mixed-backend subprocess failed in {mode!r} mode\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
