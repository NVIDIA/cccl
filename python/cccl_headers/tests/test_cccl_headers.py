# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.metadata
import importlib.util
import subprocess
import textwrap
from pathlib import Path

import pytest

import cuda.cccl.headers.include_paths as include_paths_module
from cuda import cccl
from cuda.cccl import headers


@pytest.fixture(autouse=True)
def _clear_include_path_cache():
    headers.get_include_paths.cache_clear()
    yield
    headers.get_include_paths.cache_clear()


def test_public_api():
    assert isinstance(headers.__version__, str)
    assert headers.__version__ == importlib.metadata.version("cccl-headers")
    assert headers.__all__ == [
        "IncludePaths",
        "__version__",
        "get_include_paths",
    ]
    assert cccl.__version__ == headers.__version__
    assert cccl.IncludePaths is headers.IncludePaths
    assert cccl.get_include_paths is headers.get_include_paths
    assert cccl.__all__ == headers.__all__


def test_no_legacy_top_level_package():
    assert importlib.util.find_spec("cccl_headers") is None


def test_include_paths_order():
    paths = headers.IncludePaths(
        cuda=Path("/cuda"),
        libcudacxx=Path("/libcudacxx"),
        cub=Path("/cub"),
        thrust=Path("/thrust"),
    )

    assert paths.as_tuple() == (
        paths.thrust,
        paths.cub,
        paths.libcudacxx,
        paths.cuda,
    )


def test_installed_headers_and_cmake_packages():
    paths = headers.get_include_paths()
    assert paths.cuda is not None
    assert paths.cuda.is_dir()
    assert paths.cub == paths.libcudacxx == paths.thrust
    assert paths.cub is not None

    expected_headers = (
        "cub/version.cuh",
        "thrust/version.h",
        "cuda/version",
        "nv/target",
        "cuda/experimental/coop.cuh",
        "cuda/experimental/group.cuh",
        "cuda/experimental/__multi_gpu/algorithm/common.h",
        "cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h",
        "cuda/experimental/__multi_gpu/algorithm/scan/scan.h",
        "cuda/experimental/__multi_gpu/concepts.h",
        "cuda/experimental/__multi_gpu/nccl_communicator.h",
        "cuda/experimental/__multi_gpu/nccl_communicator_ref.h",
        "cuda/experimental/__nccl/abi_compatible.h",
        "cuda/experimental/__nccl/nccl_api.h",
        "cuda/experimental/__nccl/shared_library.h",
    )
    for header in expected_headers:
        assert (paths.cub / header).is_file(), header

    package_root = paths.cub.parent
    expected_cmake_packages = (
        "cccl/cccl-config.cmake",
        "cub/cub-config.cmake",
        "cudax/cudax-config.cmake",
        "libcudacxx/libcudacxx-config.cmake",
        "thrust/thrust-config.cmake",
    )
    for package_file in expected_cmake_packages:
        assert (package_root / "lib" / "cmake" / package_file).is_file(), package_file


def test_complete_license_set():
    package_root = Path(headers.__file__).parent
    expected_licenses = (
        "licenses/LICENSE",
        "licenses/cub/LICENSE.TXT",
        "licenses/cudax/LICENSE.TXT",
        "licenses/libcudacxx/LICENSE.TXT",
        "licenses/thrust/LICENSE",
    )

    for license_file in expected_licenses:
        path = package_root / license_file
        assert path.is_file(), license_file
        assert path.stat().st_size > 1_000, license_file


def test_cudax_cmake_target_is_consumer_safe(tmp_path):
    package_root = Path(headers.__file__).parent
    cudax_cmake_dir = package_root / "lib" / "cmake" / "cudax"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "CMakeLists.txt").write_text(
        textwrap.dedent(
            """\
            cmake_minimum_required(VERSION 3.21)
            project(cccl_headers_cudax_contract LANGUAGES NONE)

            find_package(cudax CONFIG REQUIRED)

            get_target_property(cudax_alias cudax::cudax ALIASED_TARGET)
            if (NOT cudax_alias STREQUAL "_cudax_cudax")
              message(FATAL_ERROR "cudax::cudax must alias a non-imported target")
            endif()

            get_target_property(cudax_definitions _cudax_cudax INTERFACE_COMPILE_DEFINITIONS)
            if (NOT "_CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX" IN_LIST cudax_definitions)
              message(FATAL_ERROR "cudax::cudax is missing the group feature definition")
            endif()

            get_target_property(cudax_features _cudax_cudax INTERFACE_COMPILE_FEATURES)
            foreach (required_feature IN ITEMS cxx_std_17 cuda_std_17)
              if (NOT required_feature IN_LIST cudax_features)
                message(FATAL_ERROR "cudax::cudax is missing ${required_feature}")
              endif()
            endforeach()

            get_target_property(cudax_includes _cudax_cudax INTERFACE_INCLUDE_DIRECTORIES)
            file(REAL_PATH "${EXPECTED_CUDAX_INCLUDE}" expected_cudax_include)
            set(found_expected_cudax_include FALSE)
            foreach (cudax_include IN LISTS cudax_includes)
              file(REAL_PATH "${cudax_include}" resolved_cudax_include)
              if (resolved_cudax_include STREQUAL expected_cudax_include)
                set(found_expected_cudax_include TRUE)
              endif()
            endforeach()
            if (NOT found_expected_cudax_include)
              message(FATAL_ERROR
                "cudax::cudax selected '${cudax_includes}', expected wheel headers "
                "at '${EXPECTED_CUDAX_INCLUDE}'"
              )
            endif()
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(tmp_path / "build"),
            f"-Dcudax_DIR={cudax_cmake_dir}",
            f"-DEXPECTED_CUDAX_INCLUDE={(package_root / 'include').as_posix()}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_distribution_has_one_runtime_dependency():
    assert importlib.metadata.requires("cccl-headers") == ["cuda-pathfinder>=1.2.3"]


def test_missing_cuda_headers_has_actionable_error(monkeypatch):
    monkeypatch.setattr(
        include_paths_module,
        "find_nvidia_header_directory",
        lambda _library: None,
    )

    with pytest.raises(RuntimeError, match="Unable to locate CUDA include directory"):
        headers.get_include_paths()


def test_missing_cccl_headers_has_actionable_error(monkeypatch):
    monkeypatch.setattr(
        include_paths_module,
        "find_nvidia_header_directory",
        lambda _library: "/cuda/include",
    )

    with pytest.raises(RuntimeError, match="Unable to locate CCCL include directory"):
        headers.get_include_paths("not-a-real-header.cuh")
