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


def test_imported_from_installed_distribution():
    distribution_root = Path(
        importlib.metadata.distribution("cccl-headers").locate_file("")
    ).resolve()
    assert Path(headers.__file__).resolve().is_relative_to(distribution_root)


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
            project(cccl_headers_cudax_contract LANGUAGES CXX)

            # Discover CUDAX before CUDA is enabled. A repeated find after
            # enabling CUDA must add the CUDA language requirement.
            find_package(cudax CONFIG REQUIRED)
            find_package(CCCL CONFIG REQUIRED COMPONENTS cudax)

            if (ENABLE_CUDA)
              include(CheckLanguage)
              check_language(CUDA)
              if (CMAKE_CUDA_COMPILER)
                enable_language(CUDA)
                find_package(cudax CONFIG REQUIRED)
              endif()
            endif()

            add_executable(cudax_consumer main.cpp)
            target_link_libraries(cudax_consumer PRIVATE cudax::cudax)
            if (ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
              add_executable(cudax_cuda_consumer main.cu)
              target_link_libraries(cudax_cuda_consumer PRIVATE cudax::cudax)
            endif()

            get_target_property(cudax_imported cudax::cudax IMPORTED)
            get_target_property(cudax_links cudax::cudax INTERFACE_LINK_LIBRARIES)
            if (NOT cudax_imported OR NOT "_cudax_cudax" IN_LIST cudax_links)
              message(FATAL_ERROR
                "cudax::cudax must wrap the non-imported _cudax_cudax target"
              )
            endif()

            get_target_property(cudax_definitions _cudax_cudax INTERFACE_COMPILE_DEFINITIONS)
            if (NOT "_CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX" IN_LIST cudax_definitions)
              message(FATAL_ERROR "cudax::cudax is missing the group feature definition")
            endif()

            get_target_property(cudax_features _cudax_cudax INTERFACE_COMPILE_FEATURES)
            if (NOT cxx_std_17 IN_LIST cudax_features)
              message(FATAL_ERROR "cudax::cudax is missing cxx_std_17")
            endif()
            if (ENABLE_CUDA AND CMAKE_CUDA_COMPILER AND NOT cuda_std_17 IN_LIST cudax_features)
              message(FATAL_ERROR "CUDA consumers require cuda_std_17")
            elseif (NOT CMAKE_CUDA_COMPILER AND cuda_std_17 IN_LIST cudax_features)
              message(FATAL_ERROR "C++-only consumers must not require cuda_std_17")
            endif()

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

            add_library(cudax_export_consumer INTERFACE)
            target_link_libraries(cudax_export_consumer INTERFACE cudax::cudax)
            add_library(cccl_cudax_export_consumer INTERFACE)
            target_link_libraries(cccl_cudax_export_consumer INTERFACE CCCL::cudax)
            install(
              TARGETS cudax_export_consumer cccl_cudax_export_consumer
              EXPORT cudaxConsumerTargets
            )
            install(
              EXPORT cudaxConsumerTargets
              DESTINATION lib/cmake/cudax-consumer
              NAMESPACE cudax_consumer::
            )
            """
        ),
        encoding="utf-8",
    )
    (source_dir / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (source_dir / "main.cu").write_text("int main() { return 0; }\n", encoding="utf-8")

    export_source_dir = tmp_path / "export-source"
    export_source_dir.mkdir()
    (export_source_dir / "CMakeLists.txt").write_text(
        textwrap.dedent(
            """\
            cmake_minimum_required(VERSION 3.21)
            project(cccl_headers_cudax_export_contract LANGUAGES CXX)

            find_package(cudax CONFIG REQUIRED)
            find_package(CCCL CONFIG REQUIRED COMPONENTS cudax)
            include("${CUDAX_CONSUMER_TARGETS}")

            add_executable(cudax_export_consumer main.cpp)
            target_link_libraries(
              cudax_export_consumer
              PRIVATE
                cudax_consumer::cudax_export_consumer
                cudax_consumer::cccl_cudax_export_consumer
            )
            """
        ),
        encoding="utf-8",
    )
    (export_source_dir / "main.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )

    for enable_cuda in (False, True):
        build_dir = tmp_path / f"build-{enable_cuda}"
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(source_dir),
                "-B",
                str(build_dir),
                f"-DENABLE_CUDA={'ON' if enable_cuda else 'OFF'}",
                f"-Dcudax_DIR={cudax_cmake_dir}",
                f"-DCCCL_DIR={(package_root / 'lib' / 'cmake' / 'cccl').as_posix()}",
                f"-DEXPECTED_CUDAX_INCLUDE={(package_root / 'include').as_posix()}",
                f"-DCMAKE_INSTALL_PREFIX={build_dir / 'install'}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

        for command in (
            ["cmake", "--build", str(build_dir)],
            ["cmake", "--install", str(build_dir)],
        ):
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

        export_build_dir = tmp_path / f"export-build-{enable_cuda}"
        export_targets_file = (
            build_dir
            / "install"
            / "lib"
            / "cmake"
            / "cudax-consumer"
            / "cudaxConsumerTargets.cmake"
        )
        export_targets = export_targets_file.read_text(encoding="utf-8")
        assert export_targets.count('INTERFACE_LINK_LIBRARIES "cudax::cudax"') == 2
        assert "_cudax_cudax" not in export_targets
        assert "CCCL::cudax" not in export_targets

        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(export_source_dir),
                "-B",
                str(export_build_dir),
                f"-Dcudax_DIR={cudax_cmake_dir.as_posix()}",
                f"-DCCCL_DIR={(package_root / 'lib' / 'cmake' / 'cccl').as_posix()}",
                f"-DCUDAX_CONSUMER_TARGETS={export_targets_file.as_posix()}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

        result = subprocess.run(
            ["cmake", "--build", str(export_build_dir)],
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
