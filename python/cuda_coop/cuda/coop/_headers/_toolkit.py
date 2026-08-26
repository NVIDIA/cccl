# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Keep process-wide CUDA compiler libraries aligned with resolved headers.

NVRTC must match the selected CUDA headers exactly. nvJitLink consumes the
NVRTC-produced LTO-IR, so it may be a newer minor release, but it must have the
same major version and cannot be older than the headers.
"""

from __future__ import annotations

import ctypes
import os
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_CUDART_VERSION = re.compile(r"^\s*#\s*define\s+CUDART_VERSION\s+(\d+)", re.MULTILINE)
_PRELOAD_LOCK = threading.RLock()
_EXACT_LIBRARY_HANDLES: dict[str, object] = {}


@dataclass(frozen=True)
class ToolkitCompilerLibraries:
    """Actual process libraries and expected version for one CUDA header set."""

    nvrtc_path: str
    nvjitlink_path: str
    toolkit_version: tuple[int, int] | None


def _cuda_include_dirs(
    include_dirs: Iterable[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    return tuple(
        path
        for raw_path in include_dirs
        if (path := Path(raw_path).expanduser().resolve())
        .joinpath("cuda_runtime_api.h")
        .is_file()
    )


def _toolkit_version(
    cuda_include_dirs: tuple[Path, ...],
) -> tuple[int, int] | None:
    versions: set[tuple[int, int]] = set()
    for include_dir in cuda_include_dirs:
        header = include_dir / "cuda_runtime_api.h"
        try:
            contents = header.read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise RuntimeError(
                f"failed decoding CUDA Toolkit version header: {header}"
            ) from exc
        match = _CUDART_VERSION.search(contents)
        if match is None:
            continue
        encoded = int(match.group(1))
        versions.add((encoded // 1000, (encoded % 1000) // 10))
    if len(versions) > 1:
        rendered = ", ".join(f"{major}.{minor}" for major, minor in sorted(versions))
        raise RuntimeError(
            f"resolved CUDA include directories disagree on Toolkit version: {rendered}"
        )
    return next(iter(versions), None)


def _library_dirs(cuda_include_dirs: tuple[Path, ...]) -> tuple[Path, ...]:
    result: list[Path] = []
    for include_dir in cuda_include_dirs:
        for name in ("lib", "lib64", "bin"):
            candidate = include_dir.parent / name
            if candidate.is_dir() and candidate not in result:
                result.append(candidate)
    return tuple(result)


def _library_names(kind: str, major: int) -> tuple[str, ...]:
    if os.name == "nt":
        version = major * 10
        if kind == "nvrtc":
            return (f"nvrtc64_{version}_0.dll",)
        return (f"nvJitLink_{version}_0.dll",)
    if kind == "nvrtc":
        return (f"libnvrtc.so.{major}",)
    return (f"libnvJitLink.so.{major}",)


def _preload_exact_library(
    kind: str,
    *,
    major: int,
    library_dirs: tuple[Path, ...],
) -> str | None:
    """Load an exact toolkit candidate while ``_PRELOAD_LOCK`` is held."""

    failures: list[tuple[Path, OSError]] = []
    for library_dir in library_dirs:
        for name in _library_names(kind, major):
            candidate = library_dir / name
            if not candidate.is_file():
                continue
            exact_path = os.path.realpath(candidate)
            if exact_path in _EXACT_LIBRARY_HANDLES:
                return exact_path
            try:
                handle = ctypes.CDLL(
                    str(candidate),
                    mode=getattr(ctypes, "RTLD_GLOBAL", 0),
                )
            except OSError as exc:
                failures.append((candidate, exc))
                continue
            _EXACT_LIBRARY_HANDLES[exact_path] = handle
            return exact_path
    if failures:
        rendered = "; ".join(f"{candidate}: {error}" for candidate, error in failures)
        raise RuntimeError(
            f"failed loading all resolved CUDA Toolkit {kind} candidates: {rendered}"
        ) from failures[-1][1]
    return None


def _nvjitlink_version(path: str) -> tuple[int, int]:
    """Query a loaded nvJitLink while ``_PRELOAD_LOCK`` is held."""

    exact_path = os.path.realpath(path)
    if exact_path not in _EXACT_LIBRARY_HANDLES:
        try:
            handle = ctypes.CDLL(
                exact_path,
                mode=getattr(ctypes, "RTLD_GLOBAL", 0),
            )
        except OSError as exc:
            raise RuntimeError(
                f"unable to load nvJitLink for version validation: {exact_path}"
            ) from exc
        _EXACT_LIBRARY_HANDLES[exact_path] = handle
    handle = _EXACT_LIBRARY_HANDLES[exact_path]
    try:
        version = getattr(handle, "nvJitLinkVersion")
    except AttributeError as exc:
        raise RuntimeError(
            f"loaded nvJitLink does not export nvJitLinkVersion: {exact_path}"
        ) from exc
    version.argtypes = (
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    )
    version.restype = ctypes.c_int
    major = ctypes.c_uint()
    minor = ctypes.c_uint()
    result = version(ctypes.byref(major), ctypes.byref(minor))
    if result != 0:
        raise RuntimeError(
            f"nvJitLinkVersion failed with result {result}: {exact_path}"
        )
    return major.value, minor.value


def preload_toolkit_compiler_libraries(
    include_dirs: Iterable[str | os.PathLike[str]],
) -> ToolkitCompilerLibraries:
    """Preload and identify NVRTC/nvJitLink for resolved CUDA headers."""

    from cuda.pathfinder import load_nvidia_dynamic_lib

    cuda_include_dirs = _cuda_include_dirs(include_dirs)
    toolkit_version = _toolkit_version(cuda_include_dirs)
    library_dirs = _library_dirs(cuda_include_dirs)
    exact_paths: dict[str, str | None] = {"nvrtc": None, "nvJitLink": None}

    with _PRELOAD_LOCK:
        if toolkit_version is not None:
            major = toolkit_version[0]
            for kind in exact_paths:
                exact_paths[kind] = _preload_exact_library(
                    kind,
                    major=major,
                    library_dirs=library_dirs,
                )

        loaded = {
            kind: load_nvidia_dynamic_lib(kind) for kind in ("nvrtc", "nvJitLink")
        }

        actual_paths = {
            kind: os.path.realpath(result.abs_path) for kind, result in loaded.items()
        }
        for kind, exact_path in exact_paths.items():
            if exact_path is not None and actual_paths[kind] != exact_path:
                raise RuntimeError(
                    f"resolved CUDA headers expect {exact_path}, but the process uses "
                    f"{actual_paths[kind]} for {kind}"
                )

        libraries = ToolkitCompilerLibraries(
            nvrtc_path=actual_paths["nvrtc"],
            nvjitlink_path=actual_paths["nvJitLink"],
            toolkit_version=toolkit_version,
        )
        if toolkit_version is not None:
            validate_nvjitlink_version(
                libraries,
                _nvjitlink_version(libraries.nvjitlink_path),
            )
        return libraries


def validate_nvrtc_version(
    libraries: ToolkitCompilerLibraries,
    actual_version: tuple[int, int],
) -> None:
    """Require NVRTC to match the selected CUDA headers exactly."""

    expected = libraries.toolkit_version
    if expected is None or actual_version == expected:
        return
    raise RuntimeError(
        "resolved CUDA headers report Toolkit "
        f"{expected[0]}.{expected[1]}, but loaded NVRTC "
        f"{libraries.nvrtc_path} reports {actual_version[0]}.{actual_version[1]}"
    )


def validate_nvjitlink_version(
    libraries: ToolkitCompilerLibraries,
    actual_version: tuple[int, int],
) -> None:
    """Require nvJitLink to be the same major as the headers and no older."""

    expected = libraries.toolkit_version
    if expected is None or (
        actual_version[0] == expected[0] and actual_version[1] >= expected[1]
    ):
        return
    raise RuntimeError(
        "resolved CUDA headers report Toolkit "
        f"{expected[0]}.{expected[1]}, but loaded nvJitLink "
        f"{libraries.nvjitlink_path} reports {actual_version[0]}.{actual_version[1]}; "
        "nvJitLink must use the same major release and be no older than the headers"
    )


__all__ = [
    "ToolkitCompilerLibraries",
    "preload_toolkit_compiler_libraries",
    "validate_nvjitlink_version",
    "validate_nvrtc_version",
]
