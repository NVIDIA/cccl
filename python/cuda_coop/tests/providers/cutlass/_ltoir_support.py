# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared final-link qualification helpers for CUTLASS provider tests."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ...support.paths import PACKAGE_ROOT
from ...support.toolchains.cutlass import find_cuda_tool, find_ptxas_with_ptx_93

SOURCE_ROOT = PACKAGE_ROOT


def _ptxas_supports_ptx_93() -> bool:
    ptxas, _ = find_ptxas_with_ptx_93()
    return ptxas is not None


def _require_runtime() -> None:
    pytest.importorskip("cuda.coop.cutlass")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable PyTorch runtime")
    if not _ptxas_supports_ptx_93():
        pytest.skip("requires ptxas support for PTX .version 9.3")
    if find_cuda_tool("nvdisasm") is None:
        pytest.skip("requires nvdisasm to inspect the final cubin")
    if find_cuda_tool("cuobjdump") is None:
        pytest.skip("requires cuobjdump to inspect final-cubin resources")


def _configure_dump_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CUTE_DSL_KEEP", "all")
    monkeypatch.setenv("CUTE_DSL_DUMP_DIR", str(tmp_path / "dsl"))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR", str(tmp_path / "bundle"))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT", "ltoir")
    for dirname in ("dsl", "cache", "bundle"):
        (tmp_path / dirname).mkdir()


def _example_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SOURCE_ROOT}{os.pathsep}{python_path}" if python_path else str(SOURCE_ROOT)
    )
    tool_directories = {
        path.parent
        for tool in ("cuobjdump", "nvdisasm")
        if (path := find_cuda_tool(tool)) is not None
    }
    if ptxas := find_ptxas_with_ptx_93()[0]:
        tool_directories.add(ptxas.parent)
    env["PATH"] = os.pathsep.join(
        [*(str(path) for path in sorted(tool_directories)), env.get("PATH", "")]
    )
    return env


def _run_example_subprocess(
    module_name: str,
    *,
    mode: str | None = None,
    module_prefix: str = "examples.cutlass",
) -> None:
    run_call = (
        "module.run_example()" if mode is None else f"module.run_example({mode!r})"
    )
    script = f"""
import importlib
import json

module = importlib.import_module("{module_prefix}.{module_name}")
module.make_runner.cache_clear()
result = {run_call}
print(json.dumps(result, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        check=False,
        capture_output=True,
        env=_example_subprocess_env(),
        text=True,
    )
    assert result.returncode == 0, (
        f"module {module_prefix}.{module_name} failed with exit code "
        f"{result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _read_one(path_pattern: str, *, tmp_path: Path) -> str:
    matches = sorted(tmp_path.glob(path_pattern))
    assert matches, f"no artifact matched {path_pattern}"
    return matches[-1].read_text(encoding="utf-8", errors="replace")


def _find_one(path_pattern: str, *, tmp_path: Path) -> Path:
    matches = sorted(tmp_path.glob(path_pattern))
    assert matches, f"no artifact matched {path_pattern}"
    return matches[-1]


def _disassemble(cubin_path: Path) -> str:
    nvdisasm = find_cuda_tool("nvdisasm")
    assert nvdisasm is not None
    result = subprocess.run(
        [str(nvdisasm), str(cubin_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _assert_ltoir_inlined(
    *,
    tmp_path: Path,
    expected_symbols: tuple[str, ...],
    expected_sass_tokens: tuple[str, ...] = (),
) -> str:
    provider_source = _read_one(
        "bundle/cuda_coop_cutlass_bundle_*.cpp", tmp_path=tmp_path
    )
    clean_mlir = _read_one("dsl/*_clean.mlir", tmp_path=tmp_path)
    ltoir_path = _find_one("cache/*.ltoir", tmp_path=tmp_path)
    cubin_path = _find_one("dsl/*.cubin", tmp_path=tmp_path)

    assert ltoir_path.stat().st_size > 0
    assert '"link-libraries"' in clean_mlir
    assert str(ltoir_path) in clean_mlir

    for symbol in expected_symbols:
        assert symbol in provider_source
        assert f"func.call @{symbol}" in clean_mlir

    sass = _disassemble(cubin_path)
    # nvdisasm retains internal libcudacxx object symbols whose mangled names
    # include the generated bundle filename (cuda_coop_cutlass_bundle_*.cu).
    # Those data symbols are not surviving provider functions. Reject an
    # executable provider section or label instead, and check every expected
    # wrapper symbol exactly below.
    assert (
        re.search(
            r"(?m)^(?:\s*\.section\s+\.text\.|\s*\.text\.)cuda_coop_cutlass",
            sass,
        )
        is None
    )
    for symbol in expected_symbols:
        assert symbol not in sass
    for token in expected_sass_tokens:
        assert token in sass
    return sass
