# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

import cuda.coop.tile._cpp as cpp_tile
from cuda.coop.tile import (
    CppTileConfig,
    get_cpp_config,
    get_cpp_include_path,
    get_cpp_interface_path,
    get_cpp_tile_library_path,
)
from cuda.coop.tile._headers import CoopIncludePaths

from ....support.paths import REPO_ROOT

TILE_ROOT = Path(cpp_tile.__file__).resolve().parent


def test_cpp_api_header_is_available_under_public_include_spelling():
    include_path = get_cpp_include_path()
    assert (include_path / "cuda" / "coop" / "tile.cuh").is_file()


def test_cpp_config_prefers_explicit_tile_library(monkeypatch, tmp_path):
    tile_library = tmp_path / "cuda_coop_tile.tilelib"
    tile_library.touch()
    monkeypatch.delenv("CUDA_COOP_TILE_LIBRARY_VERSION", raising=False)
    monkeypatch.delenv("CUDA_COOP_TILE_INTERFACE_PATH", raising=False)
    cccl_include = tmp_path / "cccl/include"
    cuda_include = tmp_path / "cuda/include"
    cccl_include.mkdir(parents=True)
    cuda_include.mkdir(parents=True)
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(tile_library))
    monkeypatch.setattr(
        cpp_tile,
        "resolve_include_paths",
        lambda **_kwargs: CoopIncludePaths(
            cccl=(cccl_include,),
            cuda=(cuda_include,),
            origin="test",
        ),
    )

    assert get_cpp_tile_library_path() == tile_library

    config = get_cpp_config()
    assert config == CppTileConfig(
        include_path=get_cpp_include_path(),
        tile_library_path=tile_library,
        support_include_paths=(cccl_include,),
    )
    assert config.include_paths == (
        get_cpp_include_path(),
        cccl_include,
    )
    windows_host_options = (
        ("-Xcompiler=/Zc:preprocessor",) if os.name == "nt" else ()
    )
    assert config.nvcc_options() == (
        "--enable-tile",
        "--tile-preview=simt",
        "-std=c++20",
        *windows_host_options,
        f"-I{get_cpp_include_path()}",
        f"-I{cccl_include}",
        "-Xtileiras",
        f"-l={tile_library}",
    )


def test_cpp_config_discovers_packaged_tile_library(monkeypatch, tmp_path):
    tile_root = (
        tmp_path / "installed package with spaces" / "cuda" / "coop" / "tile"
    )
    tile_library = tile_root / "_native" / "cuda_coop_tile.tilelib"
    tile_library.parent.mkdir(parents=True)
    tile_library.touch()
    monkeypatch.delenv("CUDA_COOP_TILE_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(cpp_tile, "__file__", str(tile_root / "_cpp.py"))
    monkeypatch.setattr(cpp_tile, "files", lambda _package: tile_root)

    assert get_cpp_tile_library_path() == tile_library
    assert CppTileConfig(tile_root / "include", tile_library).nvcc_options()[
        -1
    ] == (f'-l="{tile_library}"')

    cuda_root = (
        tmp_path / "installed CUDA components with spaces" / "nvidia" / "cu13"
    )
    assert cpp_tile._quote_nvcc_forwarded_path(cuda_root) == f'"{cuda_root}"'


def test_tile_library_linker_input_uses_source_build_version(
    monkeypatch, tmp_path
):
    tile_library = tmp_path / "cuda_coop_tile.tilelib"
    tile_library.write_bytes(b"source-built TileLib")
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(tile_library))
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_VERSION", "source-build-17")

    assert cpp_tile._get_tile_library_linker_input() == (
        "cuda-coop-tile",
        "source-build-17",
        tile_library,
    )


def test_tile_library_linker_input_requires_source_build_version(
    monkeypatch, tmp_path
):
    tile_library = tmp_path / "cuda_coop_tile.tilelib"
    tile_library.write_bytes(b"source-built TileLib")
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(tile_library))
    monkeypatch.delenv("CUDA_COOP_TILE_LIBRARY_VERSION", raising=False)

    with pytest.raises(RuntimeError, match="CUDA_COOP_TILE_LIBRARY_VERSION"):
        cpp_tile._get_tile_library_linker_input()


def test_tile_library_linker_input_uses_matching_manifest(
    monkeypatch, tmp_path
):
    tile_library = tmp_path / "cuda_coop_tile.tilelib"
    tile_library.write_bytes(b"packaged TileLib")
    manifest = {
        "cuda_coop_tile_abi": 2,
        "distribution": "cuda-coop-tile",
        "version": "0.1.0.dev0+gabcdef",
        "tile_library": {
            "filename": tile_library.name,
            "sha256": hashlib.sha256(tile_library.read_bytes()).hexdigest(),
        },
    }
    (tmp_path / "provider-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(tile_library))

    assert cpp_tile._get_tile_library_linker_input() == (
        "cuda-coop-tile",
        manifest["version"] + "+sha256." + manifest["tile_library"]["sha256"],
        tile_library,
    )


def test_rebuilding_same_distribution_version_changes_cutile_cache_key(
    monkeypatch, tmp_path
):
    from cutile_cache._cache import cache_key

    library = tmp_path / "cuda_coop_tile.tilelib"
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(library))
    keys = []
    for content in (
        b"first build",
        b"rebuilt with another SDK",
        b"first build",
    ):
        library.write_bytes(content)
        manifest = {
            "cuda_coop_tile_abi": 2,
            "distribution": "cuda-coop-tile",
            "version": "0.1.0",
            "tile_library": {
                "filename": library.name,
                "sha256": hashlib.sha256(content).hexdigest(),
            },
        }
        (tmp_path / "provider-manifest.json").write_text(json.dumps(manifest))
        _, version, selected = cpp_tile._get_tile_library_linker_input()
        keys.append(
            cache_key(
                "compiler 13.5",
                "sm_120",
                3,
                b"unchanged kernel bytecode",
                linker_inputs=((str(selected), version),),
            )
        )
    assert keys[0] != keys[1]
    assert keys[0] == keys[2]


def test_native_install_manifest_is_usable_by_python(monkeypatch, tmp_path):
    native = tmp_path / "lib/cuda-coop-tile"
    native.mkdir(parents=True)
    library = native / "cuda_coop_tile.tilelib"
    library.write_bytes(b"standalone native TileLib")
    template = REPO_ROOT / "tile/cpp/cmake/install-provider-manifest.cmake.in"
    configured = tmp_path / "install-manifest.cmake"
    script = tmp_path / "configure.cmake"
    script.write_text(
        'set(CUDA_COOP_TILE_INSTALL_LIBRARY_DIR "lib/cuda-coop-tile")\n'
        f'configure_file("{template.as_posix()}" '
        f'"{configured.as_posix()}" @ONLY)\n'
        f'include("{configured.as_posix()}")\n'
    )
    subprocess.run(
        ["cmake", f"-DCMAKE_INSTALL_PREFIX={tmp_path}", "-P", str(script)],
        env={
            key: value for key, value in os.environ.items() if key != "DESTDIR"
        },
        capture_output=True,
        text=True,
        check=True,
    )
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(library))
    monkeypatch.delenv("CUDA_COOP_TILE_LIBRARY_VERSION", raising=False)
    distribution, version, selected = cpp_tile._get_tile_library_linker_input()
    assert distribution == "cuda-coop-tile"
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    assert version == f"native-{digest}+sha256.{digest}"
    assert selected == library


def test_tile_library_linker_input_rejects_stale_manifest(
    monkeypatch, tmp_path
):
    tile_library = tmp_path / "cuda_coop_tile.tilelib"
    tile_library.write_bytes(b"current TileLib")
    manifest = {
        "cuda_coop_tile_abi": 2,
        "distribution": "cuda-coop-tile",
        "version": "0.1.0.dev0+gstale",
        "tile_library": {
            "filename": tile_library.name,
            "sha256": hashlib.sha256(b"older TileLib").hexdigest(),
        },
    }
    (tmp_path / "provider-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(tile_library))

    with pytest.raises(
        RuntimeError, match="does not match the selected library"
    ):
        cpp_tile._get_tile_library_linker_input()


@pytest.mark.parametrize(
    ("distribution", "version", "match"),
    [
        ("another-package", "1.0", "wrong distribution"),
        ("cuda-coop-tile", "", "no usable version"),
    ],
)
def test_tile_library_linker_input_rejects_invalid_manifest_identity(
    monkeypatch, tmp_path, distribution, version, match
):
    tile_library = tmp_path / "cuda_coop_tile.tilelib"
    tile_library.write_bytes(b"packaged TileLib")
    manifest = {
        "cuda_coop_tile_abi": 2,
        "distribution": distribution,
        "version": version,
        "tile_library": {
            "filename": tile_library.name,
            "sha256": hashlib.sha256(tile_library.read_bytes()).hexdigest(),
        },
    }
    (tmp_path / "provider-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(tile_library))

    with pytest.raises(RuntimeError, match=match):
        cpp_tile._get_tile_library_linker_input()


@pytest.mark.parametrize("abi", [None, 1, "2"])
@pytest.mark.parametrize(
    "resolve",
    [
        get_cpp_tile_library_path,
        get_cpp_config,
        cpp_tile._get_tile_library_linker_input,
    ],
)
def test_tile_library_linker_input_rejects_incompatible_interface_abi(
    monkeypatch, tmp_path, abi, resolve
):
    tile_library = tmp_path / "cuda_coop_tile.tilelib"
    tile_library.write_bytes(b"packaged TileLib")
    manifest = {
        "distribution": "cuda-coop-tile",
        "version": "0.1.0.dev0+gabcdef",
        "tile_library": {
            "filename": tile_library.name,
            "sha256": hashlib.sha256(tile_library.read_bytes()).hexdigest(),
        },
    }
    if abi is not None:
        manifest["cuda_coop_tile_abi"] = abi
    (tmp_path / "provider-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(tile_library))

    with pytest.raises(RuntimeError, match="interface ABI mismatch"):
        resolve()


def test_cpp_config_leaves_simple_forwarded_paths_unquoted():
    path = Path("C:/CUDA")

    assert cpp_tile._quote_nvcc_forwarded_path(path) == str(path)


def test_cpp_config_rejects_invalid_explicit_tile_library(
    monkeypatch, tmp_path
):
    missing = tmp_path / "missing.tilelib"
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(missing))

    with pytest.raises(
        RuntimeError,
        match="CUDA_COOP_TILE_LIBRARY_PATH does not exist",
    ):
        get_cpp_tile_library_path()


def test_cpp_config_rejects_explicit_tile_library_directory(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(tmp_path))

    with pytest.raises(
        RuntimeError,
        match="CUDA_COOP_TILE_LIBRARY_PATH must point to a file",
    ):
        get_cpp_tile_library_path()


def test_cpp_config_reports_shared_build_hint_without_packaged_tilelib(
    monkeypatch,
):
    monkeypatch.delenv("CUDA_COOP_TILE_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(cpp_tile, "_packaged_tile_library_candidates", tuple)

    with pytest.raises(RuntimeError) as excinfo:
        get_cpp_tile_library_path()

    message = str(excinfo.value)
    assert "cuda.coop.tile requires its TileIR library" in message
    assert "CUDA_COOP_TILE_LIBRARY_PATH" in message


def test_cpp_config_reports_optional_interface_without_using_it(
    monkeypatch, tmp_path
):
    tile_library = tmp_path / "cuda_coop_tile.tilelib"
    tile_library.touch()
    interface = tmp_path / "cuda_coop_tile.tilebc"
    interface.touch()
    cccl_include = tmp_path / "cccl/include"
    cccl_include.mkdir(parents=True)
    monkeypatch.setenv("CUDA_COOP_TILE_LIBRARY_PATH", str(tile_library))
    monkeypatch.setenv("CUDA_COOP_TILE_INTERFACE_PATH", str(interface))
    monkeypatch.setattr(
        cpp_tile,
        "resolve_include_paths",
        lambda **_kwargs: CoopIncludePaths(
            cccl=(cccl_include,),
            cuda=(),
            origin="test",
        ),
    )

    assert get_cpp_interface_path() == interface

    config = get_cpp_config()
    assert config.interface_path == interface
    assert config.tile_library_path == tile_library
    assert config.nvcc_options()[-2:] == (
        "-Xtileiras",
        f"-l={tile_library}",
    )
    assert all(str(interface) not in option for option in config.nvcc_options())


def test_cpp_config_rejects_invalid_optional_interface(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("CUDA_COOP_TILE_INTERFACE_PATH", str(missing))

    with pytest.raises(RuntimeError, match="TileIR interface bytecode"):
        get_cpp_interface_path()
