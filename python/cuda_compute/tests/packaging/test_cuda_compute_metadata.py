# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from packaging.markers import Marker
from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]


def test_wheel_owns_only_cuda_compute_package() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    packages = metadata["tool"]["scikit-build"]["wheel"]["packages"]
    assert packages == {"cuda/compute": "cuda/compute"}
    assert not any(path.is_file() for path in (PROJECT_ROOT / "cuda/cccl").rglob("*"))


def test_compute_has_an_independent_static_version() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    project = metadata["project"]
    assert project["dependencies"] == [
        "numpy",
        "cuda-pathfinder>=1.2.3",
        "cuda-core",
        "typing_extensions",
    ]
    assert project["version"] == "1.2.0.dev0"
    assert "dynamic" not in project
    assert "dynamic-metadata" not in metadata["tool"]
    assert "setuptools_scm" not in metadata["tool"]


def test_headers_are_not_copied_into_cuda_compute_source() -> None:
    private_headers = PROJECT_ROOT / "cuda" / "compute" / "_cccl"
    assert not private_headers.exists()


def test_compute_sources_do_not_import_cuda_cccl() -> None:
    package_dir = PROJECT_ROOT / "cuda" / "compute"
    source_suffixes = {".py", ".pyi", ".pyx", ".pxi"}
    for source in package_dir.rglob("*"):
        if source.suffix in source_suffixes:
            assert "cuda.cccl" not in source.read_text(), source


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


def test_editable_include_paths_ignore_external_cuda_cccl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    external_include = tmp_path / "cuda" / "cccl" / "headers" / "include"
    (external_include / "cub").mkdir(parents=True)
    (external_include / "cub" / "version.cuh").touch()
    monkeypatch.syspath_prepend(str(tmp_path))
    module = _load_include_paths_module(monkeypatch)
    module_file = PROJECT_ROOT / "cuda" / "compute" / "_cccl_include_paths.py"

    paths = module._editable_include_paths(module_file)

    assert paths is not None
    assert external_include not in paths


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


def test_compute_extras_reference_compute_distribution() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    for requirements in metadata["project"]["optional-dependencies"].values():
        for requirement in requirements:
            assert not requirement.startswith("cuda-cccl[")


def test_test_extras_support_python_3_10() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    extras = metadata["project"]["optional-dependencies"]
    for extra in ("test-cu12", "test-cu13", "test-sysctk12", "test-sysctk13"):
        requirements = [Requirement(requirement) for requirement in extras[extra]]
        tomli_requirements = [
            requirement for requirement in requirements if requirement.name == "tomli"
        ]
        assert len(tomli_requirements) == 1
        tomli_requirement = tomli_requirements[0]
        assert not tomli_requirement.extras
        assert not tomli_requirement.specifier
        assert tomli_requirement.url is None
        assert tomli_requirement.marker == Marker('python_version < "3.11"')
        assert any(
            requirement.name == "packaging" and requirement.marker is None
            for requirement in requirements
        )


def test_pip_toolkit_extras_include_v2_windows_runtime() -> None:
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

    bindings_source = (PROJECT_ROOT / "cuda/compute/_bindings.py").read_text()
    assert 'libraries.append("nvfatbin")' in bindings_source


def test_hostjit_install_root_is_private_to_cuda_compute() -> None:
    compute_cmake = (PROJECT_ROOT / "CMakeLists.txt").read_text()
    hostjit_cmake = (
        REPOSITORY_ROOT / "c/parallel.v2/src/hostjit/CMakeLists.txt"
    ).read_text()

    setting = 'CCCL_C_PARALLEL_V2_HOSTJIT_INSTALL_ROOT "cuda/compute/_cccl"'
    assert setting in compute_cmake
    assert (
        'set(CCCL_C_PARALLEL_V2_HOSTJIT_INSTALL_ROOT "cuda/cccl/headers")'
        in hostjit_cmake
    )
    assert '"${CCCL_C_PARALLEL_V2_HOSTJIT_INSTALL_ROOT}/clang"' in hostjit_cmake
    assert '"${CCCL_C_PARALLEL_V2_HOSTJIT_INSTALL_ROOT}/libnvcc"' in hostjit_cmake
    assert (
        '"${CCCL_C_PARALLEL_V2_HOSTJIT_INSTALL_ROOT}/hostjit/cuda_minimal"'
        in hostjit_cmake
    )


def test_only_wheel_builds_install_private_headers() -> None:
    compute_cmake = (PROJECT_ROOT / "CMakeLists.txt").read_text()

    assert 'if (SKBUILD_STATE STREQUAL "wheel")' in compute_cmake
    assert 'set(CMAKE_INSTALL_INCLUDEDIR "cuda/compute/_cccl/include")' in compute_cmake
    assert 'set(CMAKE_INSTALL_LIBDIR "cuda/compute/_cccl/lib")' in compute_cmake


def test_wheel_and_editable_builds_use_separate_cmake_trees() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    assert metadata["tool"]["scikit-build"]["build-dir"] == (
        "build/{state}/{wheel_tag}"
    )


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
