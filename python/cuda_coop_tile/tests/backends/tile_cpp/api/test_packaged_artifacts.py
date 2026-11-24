# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import json

import pytest

import cuda.coop.tile._cpp as cpp_tile


@pytest.mark.parametrize("installation", ["wheel", "editable", "source"])
def test_packaged_artifacts_follow_package_resources(
    monkeypatch, tmp_path, installation
):
    source = tmp_path / "source"
    resources = source if installation == "wheel" else tmp_path / "installed"
    package = resources if installation == "editable" else source
    native = package / "_native"
    native.mkdir(parents=True)
    library = native / "cuda_coop_tile.tilelib"
    library.write_bytes(b"packaged TileLib")
    fingerprint = hashlib.sha256(library.read_bytes()).hexdigest()
    (native / "provider-manifest.json").write_text(
        json.dumps(
            {
                "cuda_coop_tile_abi": 2,
                "distribution": "cuda-coop-tile",
                "version": "0.1.0",
                "tile_library": {
                    "filename": library.name,
                    "sha256": fingerprint,
                },
            }
        )
    )
    cmake = package / "cmake"
    cmake.mkdir()
    (cmake / "cuda-coop-tile-config.cmake").touch()
    (cmake / "cuda-coop-tile-link.cmake").touch()
    include = package / "include"
    header = include / "cuda/coop/tile.cuh"
    header.parent.mkdir(parents=True)
    header.touch()
    monkeypatch.setattr(cpp_tile, "__file__", str(source / "_cpp.py"))
    monkeypatch.setattr(cpp_tile, "files", lambda _package: resources)
    for name in (
        "CUDA_COOP_TILE_LIBRARY_PATH",
        "CUDA_COOP_TILE_LIBRARY_VERSION",
        "CUDA_COOP_TILE_INTERFACE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    assert cpp_tile.get_cpp_tile_library_path() == library
    assert cpp_tile._get_tile_library_linker_input() == (
        "cuda-coop-tile",
        f"0.1.0+sha256.{fingerprint}",
        library,
    )
    assert cpp_tile.get_cmake_dir() == cmake
    assert cpp_tile.get_cpp_include_path() == include
    assert (cpp_tile.get_cmake_dir() / "cuda-coop-tile-link.cmake").is_file()
    assert cpp_tile.get_cpp_interface_path() is None

    interface = package / "_interface" / "cuda_coop_tile.tilebc"
    interface.parent.mkdir()
    interface.touch()
    assert cpp_tile.get_cpp_interface_path() == interface

    override = tmp_path / "explicit.tilelib"
    override.touch()
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(override))
    assert cpp_tile.get_cpp_tile_library_path() == override
    override.unlink()
    with pytest.raises(RuntimeError, match="does not exist"):
        cpp_tile.get_cpp_tile_library_path()

    monkeypatch.setenv("CUDA_COOP_TILE_INTERFACE_PATH", str(override))
    with pytest.raises(RuntimeError, match="TileIR interface bytecode"):
        cpp_tile.get_cpp_interface_path()


def test_foreign_call_cache_distinguishes_header_selection(
    monkeypatch, tmp_path
):
    from cutile_cache._cache import cache_key

    from cuda.coop.tile import _lowerings as lowerings

    for name in (
        "CCCL_ROOT",
        "CCCL_DIR",
        "CUDA_COOP_TILE_CUDA_ROOT",
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDA_PATH",
        "CUDA_TOOLKIT_PATH",
        "CUDACXX",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    library = tmp_path / "same.tilelib"
    monkeypatch.setattr(
        lowerings,
        "_get_tile_library_linker_input",
        lambda: ("cuda-coop-tile", "same-build", library),
    )
    encoded_paths = lowerings._PROVIDER_INCLUDE_PATH_SEPARATOR.join(
        (str(tmp_path / "first"), str(tmp_path / "second"))
    )
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS", encoded_paths)
    monkeypatch.delenv("CUDA_COOP_TILE_INCLUDE_PATHS", raising=False)
    monkeypatch.setenv("CUDA_COOP_CCCL_ROOT", str(tmp_path / "root-a"))

    def key():
        name, version, path = lowerings._coop_tile_linker_input()
        assert name == "cuda-coop-tile"
        assert path == library
        return cache_key(
            "same-compiler",
            "sm_120",
            3,
            b"same-bytecode",
            linker_inputs=((str(path), version),),
        )

    original = key()
    monkeypatch.setenv("CUDA_COOP_CCCL_ROOT", str(tmp_path / "root-b"))
    assert key() != original
    monkeypatch.setenv("CUDA_COOP_CCCL_ROOT", str(tmp_path / "root-a"))
    assert key() == original

    monkeypatch.setenv("CUDACXX", str(tmp_path / "toolkit-a/bin/nvcc"))
    compiler_key = key()
    monkeypatch.setenv("CUDACXX", str(tmp_path / "toolkit-b/bin/nvcc"))
    assert key() != compiler_key
    monkeypatch.delenv("CUDACXX")
    assert key() == original

    monkeypatch.setenv(
        "CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS",
        lowerings._PROVIDER_INCLUDE_PATH_SEPARATOR.join(
            (str(tmp_path / "second"), str(tmp_path / "first"))
        ),
    )
    assert key() != original
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS", encoded_paths)
    monkeypatch.setenv("CUDA_COOP_TILE_INCLUDE_PATHS", str(tmp_path / "extra"))
    assert key() != original
    monkeypatch.delenv("CUDA_COOP_TILE_INCLUDE_PATHS")

    other_directory = tmp_path / "other"
    other_directory.mkdir()
    with monkeypatch.context() as context:
        context.chdir(other_directory)
        assert key() == original
    for name in (
        "CUDA_COOP_CCCL_ROOT",
        "CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS",
        "CUDACXX",
    ):
        with monkeypatch.context() as context:
            context.setenv(name, "relative-headers")
            relative_key = key()
            context.chdir(other_directory)
            assert key() != relative_key
