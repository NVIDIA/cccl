# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from ...support.paths import PACKAGE_ROOT


@pytest.mark.parametrize("environment_parent", [".", "python/cuda_coop_tile"])
def test_tile_provider_resolves_its_own_installed_header_bundle(
    tmp_path, environment_parent
):
    checkout = tmp_path / "checkout"
    for directory in ("cub/cub", "thrust", "libcudacxx/include"):
        (checkout / directory).mkdir(parents=True)
    (checkout / "cub/cub/version.cuh").write_text("// ancestor CUB\n")
    toolkit = tmp_path / "toolkit"
    cuda_include = toolkit / "include"
    cuda_include.mkdir(parents=True)
    (cuda_include / "cuda_runtime.h").touch()
    site_packages = (
        checkout / environment_parent / ".venv/lib/python3.13/site-packages"
    )
    installed_headers = site_packages / "cuda" / "coop" / "tile" / "_headers"
    installed_tile = site_packages / "cuda" / "coop" / "tile"
    installed_headers.mkdir(parents=True)
    installed_tile.mkdir(parents=True, exist_ok=True)
    (installed_tile / "__init__.py").touch()
    shutil.copy2(
        PACKAGE_ROOT / "cuda" / "coop" / "tile" / "_headers" / "__init__.py",
        installed_headers / "__init__.py",
    )
    shutil.copy2(
        PACKAGE_ROOT / "cuda" / "coop" / "tile" / "_provider_runtime.py",
        installed_tile / "_provider_runtime.py",
    )
    bundled_cub = installed_tile / "include" / "cccl" / "cub"
    bundled_cub.mkdir(parents=True)
    (bundled_cub / "version.cuh").write_text(
        "// bundled CUB\n", encoding="utf-8"
    )

    script = r"""
import json
import os
import sys
from pathlib import Path

site_packages = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(site_packages))
sys.meta_path = [
    finder for finder in sys.meta_path
    if "cuda.coop.tile" not in getattr(finder, "known_source_files", {})
]
from cuda.coop.tile import _provider_runtime as runtime
from cuda.coop.tile._headers import HeaderResolutionError

assert Path(runtime.__spec__.origin).resolve() == (
    site_packages / "cuda/coop/tile/_provider_runtime.py"
)

installed = runtime._include_paths()
os.environ["CUDA_COOP_CCCL_ROOT"] = str(site_packages / "missing-root")
try:
    runtime._include_paths()
except HeaderResolutionError as error:
    assert "missing-root" in str(error)
else:
    raise AssertionError("invalid explicit CCCL root was ignored")
os.environ["CUDA_COOP_CCCL_ROOT"] = sys.argv[2]
configured = runtime._include_paths()
assert configured[:3] == [
    str(Path(sys.argv[2]) / relative)
    for relative in ("thrust", "cub", "libcudacxx/include")
]
os.environ["CUDA_COOP_TILE_CUDA_ROOT"] = str(site_packages / "missing-toolkit")
try:
    runtime._include_paths()
except HeaderResolutionError as error:
    assert "missing-toolkit" in str(error)
else:
    raise AssertionError("invalid explicit CUDA toolkit was ignored")
print(json.dumps(installed))
"""
    env = os.environ.copy()
    for name in (
        "CCCL_DIR",
        "CCCL_ROOT",
        "CUDA_COOP_CCCL_ROOT",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_PATH",
        "PYTHONPATH",
    ):
        env.pop(name, None)
    env["CUDA_COOP_TILE_CUDA_ROOT"] = str(toolkit)
    outside_checkout = tmp_path / "outside-checkout"
    outside_checkout.mkdir()

    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(site_packages), str(checkout)],
        cwd=outside_checkout,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    include_paths = json.loads(result.stdout)
    assert include_paths[0] == str(installed_tile / "include" / "cccl")
    assert str(checkout / "cub") not in include_paths
    assert include_paths[-1] == str(cuda_include)


def test_provider_runtime_rejects_conflicting_cuda_toolkits(
    monkeypatch, tmp_path
):
    from cuda.coop.tile._provider_runtime import _include_paths

    for name in ("CUDA_HOME", "CUDA_ROOT", "CUDA_PATH", "CUDA_TOOLKIT_PATH"):
        monkeypatch.delenv(name, raising=False)
    for name in ("first", "second"):
        include = tmp_path / name / "include"
        include.mkdir(parents=True)
        (include / "cuda_runtime.h").touch()
    monkeypatch.setenv("CUDA_COOP_TILE_CUDA_ROOT", str(tmp_path / "first"))
    monkeypatch.setenv("CUDA_TOOLKIT_PATH", str(tmp_path / "first/../first"))
    assert _include_paths()[-1] == str(tmp_path / "first/include")
    monkeypatch.setenv("CUDA_TOOLKIT_PATH", str(tmp_path / "second"))
    with pytest.raises(RuntimeError, match="Conflicting CUDA toolkit roots"):
        _include_paths()
