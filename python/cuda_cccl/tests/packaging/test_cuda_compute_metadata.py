# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]


def test_wheel_owns_only_cuda_compute_and_keeps_headers_out_of_source() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    packages = metadata["tool"]["scikit-build"]["wheel"]["packages"]
    assert packages == {"cuda/compute": "cuda/compute"}
    assert (PROJECT_ROOT / "cuda/compute/py.typed").is_file()
    assert not any(path.is_file() for path in (PROJECT_ROOT / "cuda/cccl").rglob("*"))
    private_headers = PROJECT_ROOT / "cuda" / "compute" / "_cccl"
    assert not private_headers.exists()


def _load_include_paths_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pathfinder = ModuleType("cuda.pathfinder")
    pathfinder.find_nvidia_header_directory = lambda _: Path("/cuda/include")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cuda.pathfinder", pathfinder)

    module_path = PROJECT_ROOT / "cuda" / "compute" / "_cccl_include_paths.py"
    spec = importlib.util.spec_from_file_location(
        "cccl_include_paths_test", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_editable_include_paths_use_canonical_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_include_paths_module(monkeypatch)
    module_file = PROJECT_ROOT / "cuda" / "compute" / "_cccl_include_paths.py"

    paths = module._editable_include_paths(module_file)

    assert paths == (
        REPOSITORY_ROOT / "libcudacxx" / "include",
        REPOSITORY_ROOT / "cub",
        REPOSITORY_ROOT / "thrust",
    )


def test_private_wheel_include_requires_all_header_families(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_include_paths_module(monkeypatch)
    include_dir = tmp_path / "_cccl" / "include"
    probes = (
        include_dir / "cub" / "version.cuh",
        include_dir / "thrust" / "version.h",
        include_dir / "cuda" / "std" / "version",
    )
    for probe in probes:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.touch()

    assert module._private_wheel_include(tmp_path) == include_dir

    probes[-1].unlink()
    assert module._private_wheel_include(tmp_path) is None


def test_pip_toolkit_extras_include_nvfatbin() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    extras = metadata["project"]["optional-dependencies"]
    for extra in ("minimal-cu12", "minimal-cu13"):
        toolkit_requirement = next(
            requirement
            for requirement in extras[extra]
            if requirement.startswith("cuda-toolkit[")
        )
        assert "nvfatbin" in toolkit_requirement.partition("]")[0].split(",")


def test_redistributed_header_licenses_are_packaged() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    force_include = metadata["tool"]["scikit-build"]["wheel"]["force-include"]
    assert force_include == {
        "../../LICENSE": "${SKBUILD_METADATA_DIR}/licenses/LICENSE",
        "../../cub/LICENSE.TXT": ("${SKBUILD_METADATA_DIR}/licenses/cub/LICENSE.TXT"),
        "../../libcudacxx/LICENSE.TXT": (
            "${SKBUILD_METADATA_DIR}/licenses/libcudacxx/LICENSE.TXT"
        ),
        "../../thrust/LICENSE": "${SKBUILD_METADATA_DIR}/licenses/thrust/LICENSE",
    }
