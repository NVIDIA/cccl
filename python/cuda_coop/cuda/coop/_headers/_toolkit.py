# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Keep process-wide CUDA compiler libraries aligned with resolved headers.

NVRTC, its builtins library, and nvJitLink are loaded from one monolithic CUDA
Toolkit root or one split-wheel ``nvidia`` anchor before a binding is allowed
to resolve either compiler library. NVRTC must match the selected CUDA headers
exactly. nvJitLink consumes NVRTC's LTO-IR, so it may be a newer minor release,
but it must have the same major version and cannot be older than the headers.
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
_PROCESS_TOOLKIT_SELECTION: tuple[str, tuple[int, int]] | None = None


@dataclass(frozen=True)
class ToolkitCompilerLibraries:
    """Actual compiler libraries and versions for one CUDA Toolkit root."""

    toolkit_root: str
    nvrtc_path: str
    nvrtc_builtins_path: str
    nvjitlink_path: str
    toolkit_version: tuple[int, int]
    nvrtc_version: tuple[int, int]
    nvjitlink_version: tuple[int, int]


@dataclass(frozen=True)
class _ToolkitRootCandidates:
    """Exact compiler-library candidates belonging to one Toolkit root."""

    toolkit_root: Path
    nvrtc_pairs: tuple[tuple[Path, Path], ...]
    nvjitlink: tuple[Path, ...]


@dataclass(frozen=True)
class _ToolkitLibraryLayout:
    """Compiler-library directories sharing one logical Toolkit root."""

    toolkit_root: Path
    nvrtc_dirs: tuple[Path, ...]
    nvjitlink_dirs: tuple[Path, ...]


def _cuda_include_dirs(
    include_dirs: Iterable[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    result: list[Path] = []
    for raw_path in include_dirs:
        path = Path(raw_path).expanduser().resolve()
        if (path / "cuda_runtime_api.h").is_file() and path not in result:
            result.append(path)
    return tuple(result)


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
            raise RuntimeError(
                "failed parsing CUDART_VERSION from CUDA Toolkit version header: "
                f"{header}"
            )
        encoded = int(match.group(1))
        versions.add((encoded // 1000, (encoded % 1000) // 10))
    if len(versions) > 1:
        rendered = ", ".join(f"{major}.{minor}" for major, minor in sorted(versions))
        raise RuntimeError(
            f"resolved CUDA include directories disagree on Toolkit version: {rendered}"
        )
    return next(iter(versions), None)


def _library_dirs(root: Path) -> tuple[Path, ...]:
    result: list[Path] = []
    for name in ("lib", "lib64", "bin"):
        candidate = root / name
        if candidate.is_dir():
            result.append(candidate)
    return tuple(result)


def _toolkit_library_layout(include_dir: Path) -> _ToolkitLibraryLayout:
    """Map CUDA headers to a monolithic or split-wheel Toolkit layout."""

    toolkit_root = include_dir.parent.resolve()
    if (
        include_dir.name == "include"
        and toolkit_root.name == "cuda_runtime"
        and toolkit_root.parent.name == "nvidia"
    ):
        # CUDA 12 wheels install Toolkit components into sibling namespace
        # packages below one ``site-packages/nvidia`` anchor. Treat that anchor
        # as the Toolkit root without searching any other site-packages tree.
        toolkit_root = toolkit_root.parent
        return _ToolkitLibraryLayout(
            toolkit_root=toolkit_root,
            nvrtc_dirs=_library_dirs(toolkit_root / "cuda_nvrtc"),
            nvjitlink_dirs=_library_dirs(toolkit_root / "nvjitlink"),
        )

    library_dirs = _library_dirs(toolkit_root)
    return _ToolkitLibraryLayout(
        toolkit_root=toolkit_root,
        nvrtc_dirs=library_dirs,
        nvjitlink_dirs=library_dirs,
    )


def _library_names(kind: str, major: int) -> tuple[str, ...]:
    if os.name == "nt":
        version = major * 10
        if kind == "nvrtc":
            return (f"nvrtc64_{version}_0.dll",)
        return (f"nvJitLink_{version}_0.dll",)
    if kind == "nvrtc":
        return (f"libnvrtc.so.{major}",)
    return (f"libnvJitLink.so.{major}",)


def _nvrtc_builtins_names(major: int, minor: int) -> tuple[str, ...]:
    if os.name == "nt":
        return (f"nvrtc-builtins64_{major}{minor}.dll",)
    return (f"libnvrtc-builtins.so.{major}.{minor}",)


def _toolkit_root_candidates(
    include_dir: Path,
    *,
    major: int,
    minor: int,
) -> tuple[_ToolkitRootCandidates | None, str]:
    """Find a complete compiler-library set below one CUDA Toolkit root."""

    layout = _toolkit_library_layout(include_dir)
    nvrtc_pairs: list[tuple[Path, Path]] = []
    unpaired_nvrtc: list[Path] = []
    for library_dir in layout.nvrtc_dirs:
        builtins = next(
            (
                library_dir / name
                for name in _nvrtc_builtins_names(major, minor)
                if (library_dir / name).is_file()
            ),
            None,
        )
        for name in _library_names("nvrtc", major):
            nvrtc = library_dir / name
            if not nvrtc.is_file():
                continue
            if builtins is None:
                unpaired_nvrtc.append(nvrtc)
            else:
                nvrtc_pairs.append((nvrtc, builtins))

    nvjitlink = tuple(
        candidate
        for library_dir in layout.nvjitlink_dirs
        for name in _library_names("nvJitLink", major)
        if (candidate := library_dir / name).is_file()
    )

    missing: list[str] = []
    if not nvrtc_pairs:
        expected_builtins = ", ".join(_nvrtc_builtins_names(major, minor))
        if unpaired_nvrtc:
            rendered = ", ".join(str(path) for path in unpaired_nvrtc)
            missing.append(
                "NVRTC candidates without adjacent builtins "
                f"({expected_builtins}): {rendered}"
            )
        else:
            expected_nvrtc = ", ".join(_library_names("nvrtc", major))
            missing.append(
                f"NVRTC and adjacent builtins ({expected_nvrtc}; {expected_builtins})"
            )
    if not nvjitlink:
        expected_nvjitlink = ", ".join(_library_names("nvJitLink", major))
        missing.append(f"nvJitLink ({expected_nvjitlink})")
    if missing:
        return None, f"{layout.toolkit_root}: missing {'; '.join(missing)}"
    return (
        _ToolkitRootCandidates(
            toolkit_root=layout.toolkit_root,
            nvrtc_pairs=tuple(nvrtc_pairs),
            nvjitlink=nvjitlink,
        ),
        "",
    )


def _claim_process_toolkit(root: Path, version: tuple[int, int]) -> None:
    """Bind process-global compiler libraries to one Toolkit root."""

    global _PROCESS_TOOLKIT_SELECTION

    selection = (str(root.resolve()), version)
    if _PROCESS_TOOLKIT_SELECTION is None:
        _PROCESS_TOOLKIT_SELECTION = selection
        return
    if _PROCESS_TOOLKIT_SELECTION == selection:
        return
    selected_root, selected_version = _PROCESS_TOOLKIT_SELECTION
    raise RuntimeError(
        "CUDA compiler libraries are already process-global from Toolkit "
        f"{selected_root} ({selected_version[0]}.{selected_version[1]}); refusing "
        f"to mix Toolkit {selection[0]} ({version[0]}.{version[1]})"
    )


def _exact_candidate_path(candidate: Path, toolkit_root: Path) -> str:
    exact_path = Path(os.path.realpath(candidate))
    try:
        exact_path.relative_to(toolkit_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"resolved CUDA Toolkit library escapes {toolkit_root}: "
            f"{candidate} -> {exact_path}"
        ) from exc
    return str(exact_path)


def _load_exact_candidate(candidate: Path, *, toolkit_root: Path) -> str:
    """Load one exact same-root library while ``_PRELOAD_LOCK`` is held."""

    exact_path = _exact_candidate_path(candidate, toolkit_root)
    if exact_path not in _EXACT_LIBRARY_HANDLES:
        handle = ctypes.CDLL(
            str(candidate),
            mode=getattr(ctypes, "RTLD_GLOBAL", 0),
        )
        _EXACT_LIBRARY_HANDLES[exact_path] = handle
    return exact_path


def _preload_toolkit_root(
    candidates: _ToolkitRootCandidates,
    *,
    version: tuple[int, int],
    builtins_failures: list[tuple[Path, OSError]],
) -> tuple[str, str, str] | None:
    """Load one complete compiler-library set without crossing Toolkit roots."""

    selected_nvrtc: tuple[str, str] | None = None
    for nvrtc, builtins in candidates.nvrtc_pairs:
        try:
            builtins_path = _load_exact_candidate(
                builtins,
                toolkit_root=candidates.toolkit_root,
            )
        except OSError as exc:
            builtins_failures.append((builtins, exc))
            continue

        # A successfully loaded builtins library is now process-global. Claim
        # its Toolkit before loading anything else so no later attempt can mix
        # roots after a partial failure.
        _claim_process_toolkit(candidates.toolkit_root, version)
        try:
            nvrtc_path = _load_exact_candidate(
                nvrtc,
                toolkit_root=candidates.toolkit_root,
            )
        except OSError as exc:
            raise RuntimeError(
                "failed loading same-root NVRTC after its builtins library was "
                f"made process-global: {nvrtc}: {exc}"
            ) from exc
        selected_nvrtc = nvrtc_path, builtins_path
        break

    if selected_nvrtc is None:
        return None

    nvjitlink_failures: list[tuple[Path, OSError]] = []
    for nvjitlink in candidates.nvjitlink:
        try:
            nvjitlink_path = _load_exact_candidate(
                nvjitlink,
                toolkit_root=candidates.toolkit_root,
            )
        except OSError as exc:
            nvjitlink_failures.append((nvjitlink, exc))
            continue
        return (*selected_nvrtc, nvjitlink_path)

    rendered = "; ".join(
        f"{candidate}: {error}" for candidate, error in nvjitlink_failures
    )
    raise RuntimeError(
        "failed loading all resolved CUDA Toolkit nvJitLink candidates below "
        f"{candidates.toolkit_root}: {rendered}"
    ) from nvjitlink_failures[-1][1]


def _exact_library_handle(path: str, *, kind: str) -> object:
    exact_path = os.path.realpath(path)
    try:
        return _EXACT_LIBRARY_HANDLES[exact_path]
    except KeyError as exc:
        raise RuntimeError(
            f"the exact {kind} handle was not retained for validation: {exact_path}"
        ) from exc


def _nvrtc_version(path: str) -> tuple[int, int]:
    """Query the exact loaded NVRTC while ``_PRELOAD_LOCK`` is held."""

    handle = _exact_library_handle(path, kind="NVRTC")
    try:
        version = getattr(handle, "nvrtcVersion")
    except AttributeError as exc:
        raise RuntimeError(
            f"loaded NVRTC does not export nvrtcVersion: {path}"
        ) from exc
    version.argtypes = (
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    )
    version.restype = ctypes.c_int
    major = ctypes.c_int()
    minor = ctypes.c_int()
    result = version(ctypes.byref(major), ctypes.byref(minor))
    if result != 0:
        raise RuntimeError(f"nvrtcVersion failed with result {result}: {path}")
    return major.value, minor.value


def _nvjitlink_version(path: str) -> tuple[int, int]:
    """Query the exact loaded nvJitLink while ``_PRELOAD_LOCK`` is held."""

    handle = _exact_library_handle(path, kind="nvJitLink")
    try:
        version = getattr(handle, "nvJitLinkVersion")
    except AttributeError as exc:
        raise RuntimeError(
            f"loaded nvJitLink does not export nvJitLinkVersion: {path}"
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
        raise RuntimeError(f"nvJitLinkVersion failed with result {result}: {path}")
    return major.value, minor.value


def preload_toolkit_compiler_libraries(
    include_dirs: Iterable[str | os.PathLike[str]],
) -> ToolkitCompilerLibraries:
    """Preload and validate one same-root NVRTC/builtins/nvJitLink set."""

    # Keep CUDA bindings out of root import. Pathfinder is loaded only once a
    # provider actually needs compiler libraries.
    from cuda.pathfinder import load_nvidia_dynamic_lib

    cuda_include_dirs = _cuda_include_dirs(include_dirs)
    if not cuda_include_dirs:
        raise RuntimeError(
            "no resolved include directory contains cuda_runtime_api.h; "
            "cannot select same-root CUDA compiler libraries"
        )
    toolkit_version = _toolkit_version(cuda_include_dirs)
    assert toolkit_version is not None
    major, minor = toolkit_version

    # The first CUDA header root is authoritative because it is the one the
    # compiler's ordered include search will use. A complete library set under
    # a later include root must not silently replace it, even at the same
    # reported Toolkit version.
    primary_include = cuda_include_dirs[0]
    primary_root = _toolkit_library_layout(primary_include).toolkit_root
    candidate, diagnostic = _toolkit_root_candidates(
        primary_include,
        major=major,
        minor=minor,
    )
    if candidate is None:
        other_roots = tuple(
            _toolkit_library_layout(include_dir).toolkit_root
            for include_dir in cuda_include_dirs[1:]
            if _toolkit_library_layout(include_dir).toolkit_root != primary_root
        )
        ignored = (
            "; later ordered CUDA header roots are not eligible: "
            + ", ".join(str(root) for root in other_roots)
            if other_roots
            else ""
        )
        raise RuntimeError(
            "resolved CUDA headers require NVRTC, nvrtc-builtins, and nvJitLink "
            f"from one CUDA Toolkit root: {diagnostic}{ignored}"
        )
    candidates = [candidate]

    with _PRELOAD_LOCK:
        if _PROCESS_TOOLKIT_SELECTION is not None:
            selected_root, selected_version = _PROCESS_TOOLKIT_SELECTION
            if selected_version != toolkit_version:
                raise RuntimeError(
                    "CUDA compiler libraries are already process-global from "
                    f"Toolkit {selected_root} ({selected_version[0]}."
                    f"{selected_version[1]}); resolved headers report "
                    f"{major}.{minor}"
                )
            candidates = [
                candidate
                for candidate in candidates
                if str(candidate.toolkit_root) == selected_root
            ]
            if not candidates:
                raise RuntimeError(
                    "CUDA compiler libraries are already process-global from "
                    f"Toolkit {selected_root}; resolved headers select different "
                    "Toolkit roots"
                )

        builtins_failures: list[tuple[Path, OSError]] = []
        selected: tuple[_ToolkitRootCandidates, tuple[str, str, str]] | None = None
        for candidate in candidates:
            paths = _preload_toolkit_root(
                candidate,
                version=toolkit_version,
                builtins_failures=builtins_failures,
            )
            if paths is not None:
                selected = candidate, paths
                break
        if selected is None:
            rendered = "; ".join(
                f"{candidate}: {error}" for candidate, error in builtins_failures
            )
            raise RuntimeError(
                "failed loading all same-root CUDA Toolkit NVRTC builtins "
                f"candidates: {rendered}"
            ) from builtins_failures[-1][1]

        root_candidates, (nvrtc_path, builtins_path, nvjitlink_path) = selected
        exact_paths = {
            "nvrtc": nvrtc_path,
            "nvJitLink": nvjitlink_path,
        }
        loaded = {
            kind: load_nvidia_dynamic_lib(kind) for kind in ("nvrtc", "nvJitLink")
        }
        actual_paths = {
            kind: os.path.realpath(result.abs_path) for kind, result in loaded.items()
        }
        for kind, exact_path in exact_paths.items():
            if actual_paths[kind] != exact_path:
                raise RuntimeError(
                    f"resolved CUDA headers expect {exact_path}, but the process uses "
                    f"{actual_paths[kind]} for {kind}"
                )

        nvrtc_version = _nvrtc_version(nvrtc_path)
        nvjitlink_version = _nvjitlink_version(nvjitlink_path)
        libraries = ToolkitCompilerLibraries(
            toolkit_root=str(root_candidates.toolkit_root),
            nvrtc_path=nvrtc_path,
            nvrtc_builtins_path=builtins_path,
            nvjitlink_path=nvjitlink_path,
            toolkit_version=toolkit_version,
            nvrtc_version=nvrtc_version,
            nvjitlink_version=nvjitlink_version,
        )
        validate_nvrtc_version(libraries, nvrtc_version)
        validate_nvjitlink_version(libraries, nvjitlink_version)
        return libraries


def validate_nvrtc_version(
    libraries: ToolkitCompilerLibraries,
    actual_version: tuple[int, int],
) -> None:
    """Require NVRTC to match the selected CUDA headers exactly."""

    expected = libraries.toolkit_version
    if actual_version == expected:
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
    if actual_version[0] == expected[0] and actual_version[1] >= expected[1]:
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
