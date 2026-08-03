# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import tomllib

PROJECT_ROOT = Path(__file__).parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]


def test_wheel_owns_only_cuda_compute_package() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    packages = metadata["tool"]["scikit-build"]["wheel"]["packages"]
    assert packages == {"cuda/compute": "cuda/compute"}
    assert not any(path.is_file() for path in (PROJECT_ROOT / "cuda/cccl").rglob("*"))


def test_exact_cccl_headers_dependency_follows_cuda_compute_version() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    project = metadata["project"]
    providers = metadata["tool"]["dynamic-metadata"]

    assert project["dependencies"] == [
        "numpy",
        "cuda-pathfinder>=1.2.3",
        "cuda-core",
        "typing_extensions",
    ]
    assert project["dynamic"] == ["version", "dependencies"]
    assert providers == [
        {"provider": "scikit_build_core.metadata.setuptools_scm"},
        {
            "provider": "scikit_build_core.metadata.template",
            "field": "dependencies",
            "result": ["cccl-headers=={project[version]}"],
        },
    ]


def test_compute_extras_reference_compute_distribution() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    for requirements in metadata["project"]["optional-dependencies"].values():
        for requirement in requirements:
            assert not requirement.startswith("cuda-cccl[")


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

    setting = 'CCCL_C_PARALLEL_V2_HOSTJIT_INSTALL_ROOT "cuda/compute/_hostjit"'
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


def test_hostjit_build_paths_are_disabled_only_for_wheels() -> None:
    compute_cmake = (PROJECT_ROOT / "CMakeLists.txt").read_text()
    parallel_cmake = (REPOSITORY_ROOT / "c/parallel.v2/CMakeLists.txt").read_text()
    hostjit_cmake = (
        REPOSITORY_ROOT / "c/parallel.v2/src/hostjit/CMakeLists.txt"
    ).read_text()
    libnvcc_cmake = (
        REPOSITORY_ROOT / "c/parallel.v2/src/hostjit/libnvcc/CMakeLists.txt"
    ).read_text()

    runtime_paths = "CCCL_C_PARALLEL_V2_HOSTJIT_USE_RUNTIME_PATHS"
    assert (
        f'option(\n  {runtime_paths}\n  "Resolve HostJIT resource paths at runtime '
        'instead of embedding build-tree defaults."\n  OFF\n)' in parallel_cmake
    )
    assert (
        f'if (SKBUILD_STATE STREQUAL "wheel")\n  set({runtime_paths} ON)\nendif()'
        in compute_cmake
    )
    assert f"if (NOT {runtime_paths})" in hostjit_cmake
    assert f"if (NOT {runtime_paths})" in libnvcc_cmake


def test_cython_generated_sources_use_logical_package_paths() -> None:
    compute_cmake = (PROJECT_ROOT / "CMakeLists.txt").read_text()

    assert 'set(CYTHON_FLAGS -3 -M -w "${cuda_compute_BINARY_DIR}")' in compute_cmake
    assert (
        'set(_cython_package_dir "${cuda_compute_BINARY_DIR}/cuda/compute")'
        in compute_cmake
    )
    assert '"cuda/compute/_bindings_impl.pyx"' in compute_cmake
    assert '-I "${CMAKE_CURRENT_BINARY_DIR}"' not in compute_cmake
