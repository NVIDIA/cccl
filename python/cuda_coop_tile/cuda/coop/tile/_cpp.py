# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve the installed headers and TileLib used by both Tile frontends.

The host helpers work without importing ``cuda.tile``. Explicit environment
overrides select development artifacts; otherwise resolution stays within
the package. Python foreign calls also use this resolver, then
check the library's version and content identity for compiler caching.
The optional declaration bytecode is a development artifact and is not an
extra linker input in ``CppTileConfig.nvcc_options``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from cuda.coop.tile._headers import resolve_include_paths

_TILE_LIBRARY_FILENAME = "cuda_coop_tile.tilelib"
_TILE_LIBRARY_LINKER_INPUT_NAME = "cuda-coop-tile"
_TILE_LIBRARY_MANIFEST_FILENAME = "provider-manifest.json"
_TILE_LIBRARY_VERSION_ENV = "CUDA_COOP_TILE_LIBRARY_VERSION"
_TILE_INTERFACE_ABI_VERSION = 2
_INTERFACE_FILENAME = "cuda_coop_tile.tilebc"


def _packaged_artifact_path(directory: str, filename: str) -> Path:
    resource = files("cuda.coop.tile").joinpath(directory).joinpath(filename)
    # Editable loaders can place native artifacts beside the installed package
    # while redirecting Python modules to the source checkout. Keep the native
    # file beside its manifest, provider and CMake configuration companions.
    if isinstance(resource, Path) and resource.is_file():
        return resource.absolute()
    return Path(__file__).resolve().parent / directory / filename


def _quote_nvcc_forwarded_path(path: Path) -> str:
    """Quote a path that NVCC reparses while forwarding an option."""

    value = str(path)
    if '"' in value:
        raise RuntimeError(
            f"NVCC forwarded paths must not contain a double quote: {value}"
        )
    return (
        f'"{value}"'
        if any(character.isspace() for character in value)
        else value
    )


@dataclass(frozen=True)
class CppTileConfig:
    """C++ Tile include roots and the TileLib to pass to NVCC.

    Parameters
    ----------
    include_path : pathlib.Path
        Root containing the public ``cuda/coop/tile.cuh`` header.
    tile_library_path : pathlib.Path
        TileLib containing the cooperative declarations and linkage provider.
    support_include_paths : tuple[pathlib.Path, ...]
        Additional CCCL include roots, in search order. Defaults to empty.
    interface_path : pathlib.Path or None
        Optional development declaration bytecode. It is recorded for
        inspection and is not passed to NVCC; the TileLib owns declarations.
        Defaults to ``None``.

    Notes
    -----
    This frozen configuration stores paths without validating manually
    supplied files. Use ``get_cpp_config()`` to resolve installed artifacts
    and apply its existence and manifest checks.

    Examples
    --------
    Append the returned arguments to an NVCC command as separate elements:

    .. code-block:: python

        config = coop.get_cpp_config()
        command = ["nvcc", *config.nvcc_options(), "kernel.cu", "-o", "kernel"]
    """

    include_path: Path
    tile_library_path: Path
    support_include_paths: tuple[Path, ...] = ()
    interface_path: Path | None = None

    @property
    def include_paths(self) -> tuple[Path, ...]:
        """Return the public and supporting include roots in search order.

        The public root is first. Duplicate paths are removed while keeping
        their first occurrence. This property does not check file existence.
        """

        return tuple(
            dict.fromkeys((self.include_path, *self.support_include_paths))
        )

    def nvcc_options(self) -> tuple[str, ...]:
        """Return NVCC arguments for these headers and this TileLib.

        Returns
        -------
        tuple[str, ...]
            Separate command arguments enabling Tile SIMT preview, C++20,
            include roots, and the TileLib linker input. On Windows, includes
            the host preprocessor option required by the public headers.

        Raises
        ------
        RuntimeError
            If the TileLib path contains a double quote and cannot be
            forwarded safely through NVCC to TileIRAS.

        Notes
        -----
        Pass the tuple as arguments to a process API such as ``subprocess.run``.
        It is not a shell command. Add the architecture, source, output, and
        any other build options required by the application. The optional
        ``interface_path`` is not a separate linker input.
        """

        windows_host_options = (
            ("-Xcompiler=/Zc:preprocessor",) if os.name == "nt" else ()
        )
        return (
            "--enable-tile",
            "--tile-preview=simt",
            "-std=c++20",
            *windows_host_options,
            *(f"-I{path}" for path in self.include_paths),
            "-Xtileiras",
            f"-l={_quote_nvcc_forwarded_path(self.tile_library_path)}",
        )


def get_cpp_include_path() -> Path:
    """Return the installed include root for ``cuda/coop/tile.cuh``.

    Returns
    -------
    pathlib.Path
        Include directory owned by ``cuda-coop-tile``. This root contains
        the cooperative Tile header, not the supporting CCCL headers.

    Raises
    ------
    RuntimeError
        If the public header is missing from the package.

    See Also
    --------
    get_cpp_config
    """

    include_path = _packaged_artifact_path(
        "include", "cuda/coop/tile.cuh"
    ).parents[2]
    for parent in Path(__file__).resolve().parents:
        source_header = parent / "tile/cpp/include/cuda/coop/tile.cuh"
        if (parent / "python/cuda_coop_tile/cuda/coop/tile/_cpp.py") == Path(
            __file__
        ).resolve() and source_header.is_file():
            include_path = parent / "tile/cpp/include"
            break
    header = include_path / "cuda" / "coop" / "tile.cuh"
    if not header.is_file():
        raise RuntimeError(
            "cuda-coop-tile is missing its C++ Tile header: "
            f"{header}. Install cuda-coop-tile "
            "with C++ Tile support."
        )
    return include_path


def get_cmake_dir() -> Path:
    """Return the installed ``cuda-coop-tile`` CMake package directory.

    Returns
    -------
    pathlib.Path
        Directory containing ``cuda-coop-tile-config.cmake``. Pass it as
        ``cuda-coop-tile_DIR`` when configuring a CMake consumer.

    Raises
    ------
    RuntimeError
        If the CMake configuration file is missing. Install matching
        ``cuda-coop-tile``.

    Examples
    --------
    Resolve the package directory without importing the Python compiler:

    .. code-block:: python

        import cuda.coop.tile as coop

        cmake_option = f"-Dcuda-coop-tile_DIR={coop.get_cmake_dir()}"
    """

    config = _packaged_artifact_path("cmake", "cuda-coop-tile-config.cmake")
    if not config.is_file():
        raise RuntimeError(
            "cuda-coop-tile is missing its CMake package. "
            "Install cuda-coop-tile wheels."
        )
    return config.parent


def _packaged_tile_library_candidates() -> tuple[Path, ...]:
    return (_packaged_artifact_path("_native", _TILE_LIBRARY_FILENAME),)


def _tile_library_path_diagnostics() -> str:
    return (
        "Install cuda-coop-tile with its native TileLib, or set "
        "CUDA_COOP_TILE_LIBRARY_PATH to an explicitly built "
        "cuda_coop_tile.tilelib. Native builds use CMake's tile/ project "
        "with CUDA 13.5 and the separately supplied preview SDK."
    )


def _read_tile_library_manifest(tile_library_path: Path) -> dict | None:
    manifest_path = tile_library_path.parent / _TILE_LIBRARY_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cuda.coop TileLib manifest is unreadable: {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        # All malformed artifact manifests use the same public error type.
        raise RuntimeError(  # noqa: TRY004
            "cuda.coop TileLib manifest must contain a JSON object: "
            f"{manifest_path}"
        )

    interface_abi = manifest.get("cuda_coop_tile_abi")
    if (
        type(interface_abi) is not int
        or interface_abi != _TILE_INTERFACE_ABI_VERSION
    ):
        raise RuntimeError(
            "cuda.coop TileLib interface ABI mismatch: "
            f"expected {_TILE_INTERFACE_ABI_VERSION}, got {interface_abi!r}. "
            "Rebuild the TileLib and install matching "
            "cuda-coop-tile wheels."
        )
    return manifest


def get_cpp_tile_library_path() -> Path:
    """Return the selected TileLib for C++ and Python Tile calls.

    ``CUDA_COOP_TILE_LIBRARY_PATH`` takes precedence over the library in the
    package. An explicit path expands ``~`` and resolves to an
    absolute path. A missing or invalid override raises instead of falling
    back to the packaged library.

    Returns
    -------
    pathlib.Path
        Path to the selected ``cuda_coop_tile.tilelib`` or explicit override.
        If an adjacent provider manifest exists, its interface ABI must
        match the frontend.

    Raises
    ------
    RuntimeError
        If no library exists, an override is not a file, or the adjacent
        manifest is unreadable or has an incompatible interface ABI.

    Notes
    -----
    This helper checks the path and manifest ABI. Python compilation also
    checks the manifest's library fingerprint and version. A development
    TileLib without a manifest requires ``CUDA_COOP_TILE_LIBRARY_VERSION``
    for Python calls; this path helper alone does not require that version.
    """

    explicit = os.environ.get("CUDA_COOP_TILE_LIBRARY_PATH")
    if explicit:
        tile_library = Path(explicit).expanduser().resolve()
        if not tile_library.exists():
            raise RuntimeError(
                "CUDA_COOP_TILE_LIBRARY_PATH does not exist: "
                f"{tile_library}\n{_tile_library_path_diagnostics()}"
            )
        if not tile_library.is_file():
            raise RuntimeError(
                "CUDA_COOP_TILE_LIBRARY_PATH must point to a file: "
                f"{tile_library}\n{_tile_library_path_diagnostics()}"
            )
        _read_tile_library_manifest(tile_library)
        return tile_library

    # Resolve the owning package's resources for both ordinary wheels and
    # editable installations, whose Python sources may live elsewhere.
    for tile_library in _packaged_tile_library_candidates():
        if tile_library.is_file():
            _read_tile_library_manifest(tile_library)
            return tile_library

    raise RuntimeError(
        "cuda.coop.tile requires its TileIR library, but no packaged TileLib "
        f"was found.\n{_tile_library_path_diagnostics()}"
    )


@lru_cache(maxsize=16)
def _file_sha256(path: Path, mtime_ns: int, size: int) -> str:
    del mtime_ns, size
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tile_library_sha256(tile_library_path: Path) -> str:
    stat = tile_library_path.stat()
    return _file_sha256(tile_library_path, stat.st_mtime_ns, stat.st_size)


def _get_tile_library_linker_input() -> tuple[str, str, Path]:
    """Return the stable identity and path used by preview foreign calls."""

    tile_library_path = get_cpp_tile_library_path()
    manifest_path = tile_library_path.parent / _TILE_LIBRARY_MANIFEST_FILENAME
    manifest = _read_tile_library_manifest(tile_library_path)
    if manifest is None:
        version = os.environ.get(_TILE_LIBRARY_VERSION_ENV, "").strip()
        if not version:
            raise RuntimeError(
                "Python preview calls using a TileLib without an adjacent "
                f"{_TILE_LIBRARY_MANIFEST_FILENAME} require "
                f"{_TILE_LIBRARY_VERSION_ENV}. Set it to a nonempty version "
                "and increment it whenever the library or one of its "
                "dependencies changes."
            )
        return _TILE_LIBRARY_LINKER_INPUT_NAME, version, tile_library_path

    try:
        tile_library_sha256 = _tile_library_sha256(tile_library_path)
    except OSError as exc:
        raise RuntimeError(
            "cuda.coop could not fingerprint its TileLib: "
            f"{tile_library_path}: {exc}"
        ) from exc

    distribution = manifest.get("distribution")
    version = manifest.get("version")
    tile_library = manifest.get("tile_library")
    expected_tile_library = {
        "filename": tile_library_path.name,
        "sha256": tile_library_sha256,
    }
    if distribution != _TILE_LIBRARY_LINKER_INPUT_NAME:
        raise RuntimeError(
            "cuda.coop TileLib manifest has the wrong distribution: "
            f"expected {_TILE_LIBRARY_LINKER_INPUT_NAME!r}, "
            f"got {distribution!r}"
        )
    if not isinstance(version, str) or not version:
        raise RuntimeError(
            f"cuda.coop TileLib manifest has no usable version: {manifest_path}"
        )
    if tile_library != expected_tile_library:
        raise RuntimeError(
            "cuda.coop TileLib manifest does not match the selected library: "
            f"expected {expected_tile_library!r}, got {tile_library!r}"
        )
    # cuTile keys its cubin cache by library path and version, not file bytes.
    # Rebuilding one distribution version must invalidate the previous cubin.
    cache_version = f"{version}+sha256.{tile_library_sha256}"
    return distribution, cache_version, tile_library_path


def _get_optional_cpp_artifact(
    *,
    env_name: str,
    packaged_filename: str,
    artifact_name: str,
) -> Path | None:
    explicit = os.environ.get(env_name)
    if explicit:
        artifact = Path(explicit).expanduser().resolve()
        if not artifact.is_file():
            raise RuntimeError(
                f"{env_name} must name a {artifact_name} file: {artifact}"
            )
        return artifact

    packaged = _packaged_artifact_path("_interface", packaged_filename)
    return packaged if packaged.is_file() else None


def get_cpp_interface_path() -> Path | None:
    """Return optional development bytecode for the Tile declarations.

    ``CUDA_COOP_TILE_INTERFACE_PATH`` selects an explicit file. Otherwise,
    look for ``cuda_coop_tile.tilebc`` in the package's interface
    directory. The TileLib already contains the declarations needed for
    compilation; this bytecode is useful for inspecting or developing them.

    Returns
    -------
    pathlib.Path or None
        Selected bytecode file, or ``None`` when no override or packaged
        bytecode exists. Explicit paths expand ``~`` and resolve absolutely.

    Raises
    ------
    RuntimeError
        If an explicit override does not name a file.
    """

    return _get_optional_cpp_artifact(
        env_name="CUDA_COOP_TILE_INTERFACE_PATH",
        packaged_filename=_INTERFACE_FILENAME,
        artifact_name="TileIR interface bytecode",
    )


def get_cpp_config() -> CppTileConfig:
    """Resolve the headers and TileLib for a C++ Tile build.

    Resolve supporting CCCL headers from ``CUDA_COOP_CCCL_ROOT``, a source
    checkout, or the installed header bundle. Resolve the cooperative header,
    TileLib, and optional interface through the corresponding host helpers.
    The configuration does not import cuTile or run a compiler.

    Returns
    -------
    CppTileConfig
        Resolved include roots, TileLib path, and optional declaration
        bytecode. ``nvcc_options()`` converts it to NVCC arguments.

    Raises
    ------
    RuntimeError
        If required headers or the TileLib are missing, the TileLib manifest
        is incompatible, or an explicit artifact override is invalid.

    See Also
    --------
    get_cpp_include_path, get_cpp_tile_library_path, get_cpp_interface_path
    """

    support_include_paths = resolve_include_paths(
        start=Path(__file__),
        configured_roots=(os.environ.get("CUDA_COOP_CCCL_ROOT"),),
        required_headers=(
            "cuda/functional",
            "cuda/std/functional",
            "cuda/std/type_traits",
        ),
    ).cccl
    return CppTileConfig(
        include_path=get_cpp_include_path(),
        tile_library_path=get_cpp_tile_library_path(),
        support_include_paths=support_include_paths,
        interface_path=get_cpp_interface_path(),
    )


__all__ = [
    "CppTileConfig",
    "get_cmake_dir",
    "get_cpp_config",
    "get_cpp_include_path",
    "get_cpp_interface_path",
    "get_cpp_tile_library_path",
]
