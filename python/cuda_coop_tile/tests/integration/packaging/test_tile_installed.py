# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise installed artifacts without importing the checkout's package."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ...support.paths import REPO_ROOT


def _installed_root():
    value = os.environ.get("CUDA_COOP_TILE_INSTALLED_ROOT")
    if not value:
        pytest.fail(
            "set CUDA_COOP_TILE_INSTALLED_ROOT to the installed wheel root"
        )
    root = Path(value).resolve()
    assert (root / "cuda/coop/tile/__init__.py").is_file()
    return root


def _environment():
    environment = os.environ.copy()
    for name in (
        "PYTHONPATH",
        "CUDA_COOP_TILE_LIBRARY_PATH",
        "CUDA_COOP_TILE_LIBRARY_VERSION",
        "CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS",
        "CUDA_COOP_TILE_INCLUDE_PATHS",
        "CUDA_COOP_CCCL_ROOT",
        "CCCL_ROOT",
        "CCCL_DIR",
    ):
        environment.pop(name, None)
    return environment


def _run(command, tmp_path):
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_installed_helpers_validate_payload_identity(tmp_path):
    root = _installed_root()
    script = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.meta_path = [
    finder for finder in sys.meta_path
    if "cuda.coop.tile" not in getattr(finder, "known_source_files", {})
]
import cuda.coop
import cuda.coop.tile as coop
from cuda.coop.tile._cpp import _get_tile_library_linker_input
from cuda.coop.tile._headers import resolve_include_paths
root = Path(sys.argv[1])
assert cuda.coop.__file__ is None
assert Path(coop.__file__).is_relative_to(root)
assert 'cuda.tile' not in sys.modules
identity = _get_tile_library_linker_input()
config = coop.get_cpp_config()
assert identity[0] == 'cuda-coop-tile'
assert identity[2] == root / 'cuda/coop/tile/_native/cuda_coop_tile.tilelib'
assert config.include_path == root / 'cuda/coop/tile/include'
headers = resolve_include_paths(start=Path(coop.__file__))
assert headers.cccl == (root / 'cuda/coop/tile/include/cccl',)
assert coop.get_cmake_dir().is_relative_to(root)
print(json.dumps({'version': identity[1], 'tilelib': str(identity[2])}))
"""
    receipt = json.loads(
        _run([sys.executable, "-I", "-c", script, str(root)], tmp_path)
    )
    assert receipt["version"]


@pytest.mark.gpu
@pytest.mark.runtime
def test_installed_python_executes_scan_and_reduce(tmp_path):
    root = _installed_root()
    script = tmp_path / "installed_scan.py"
    script.write_text("""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.meta_path = [
    finder for finder in sys.meta_path
    if "cuda.coop.tile" not in getattr(finder, "known_source_files", {})
]
import torch
import cuda.tile as ct
import cuda.coop.tile as coop
import cuda.coop.tile._cpp as cpp

package = Path(sys.argv[1]).resolve() / "cuda/coop/tile"
assert Path(coop.__spec__.origin).resolve() == package / "__init__.py"
assert Path(cpp.__spec__.origin).resolve() == package / "_cpp.py"

@ct.kernel
def kernel(source, scanned, summed):
    values = ct.load(source, (0,), (128,))
    ct.store(scanned, (0,), coop.exclusive_sum(values))
    ct.store(summed, (0,), coop.sum(values, broadcast=True))

source = torch.arange(128, device="cuda", dtype=torch.int32)
scanned = torch.empty_like(source)
summed = torch.empty_like(source)
ct.launch(torch.cuda.current_stream(), (1,), kernel, (source, scanned, summed))
torch.cuda.synchronize()
assert torch.equal(scanned, torch.cumsum(source, 0) - source)
assert torch.equal(summed, torch.full_like(source, source.sum()))
assert torch.equal(source, torch.arange(128, device="cuda", dtype=torch.int32))
""")
    _run([sys.executable, "-I", str(script), str(root)], tmp_path)


@pytest.mark.gpu
@pytest.mark.runtime
def test_installed_cmake_executes_cpp_scan_and_reduce(tmp_path):
    root = _installed_root()
    source = REPO_ROOT / "tile/cpp/test/support/tile_cpp_scan_reduce.cu"
    project = tmp_path / "consumer"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25.2)\n"
        "project(consumer LANGUAGES CXX CUDA)\n"
        "find_package(cuda-coop-tile CONFIG REQUIRED)\n"
        f'add_executable(consumer "{source.as_posix()}")\n'
        "cuda_coop_tile_link(consumer)\n"
    )
    build = tmp_path / "build"
    _run(
        [
            "cmake",
            "-S",
            str(project),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-Dcuda-coop-tile_DIR={root / 'cuda/coop/tile/cmake'}",
            "-DCMAKE_CUDA_ARCHITECTURES=120",
        ],
        tmp_path,
    )
    _run(["cmake", "--build", str(build)], tmp_path)
    _run(
        [str(build / ("consumer.exe" if os.name == "nt" else "consumer"))],
        tmp_path,
    )
