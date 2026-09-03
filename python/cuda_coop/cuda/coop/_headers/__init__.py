# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve the CCCL headers used to compile cooperative primitives."""

from __future__ import annotations

import atexit
import os
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import cache
from importlib.resources import as_file, files
from pathlib import Path

_CUB_PROBE = Path("cub/version.cuh")
_CUDA_PROBE = Path("cuda_runtime.h")
_INSTALLED_HEADER_CONTEXTS = ExitStack()
atexit.register(_INSTALLED_HEADER_CONTEXTS.close)


class HeaderResolutionError(RuntimeError):
    """Raised when cuda.coop cannot resolve a coherent CCCL header set."""


@dataclass(frozen=True)
class CoopIncludePaths:
    """Ordered CCCL and CUDA include paths with their provenance."""

    cccl: tuple[Path, ...]
    cuda: tuple[Path, ...]
    origin: str

    def as_tuple(self) -> tuple[Path, ...]:
        if not self.cuda:
            raise HeaderResolutionError(
                "Unable to locate one CUDA include directory containing "
                "cuda_runtime.h. Configure CUDA_PATH or CUDA_HOME before "
                "compiling cuda.coop providers."
            )
        return (*self.cccl, *self.cuda)


def _unique_existing_dirs(paths: Iterable[Path | str | None]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir() or path in seen:
            continue
        seen.add(path)
        result.append(path)
    return tuple(result)


def _source_checkout_paths(root: Path) -> tuple[Path, ...]:
    if not (root / "cub" / _CUB_PROBE).is_file():
        return ()
    candidates = (
        root / "thrust",
        root / "cub",
        root / "cudax" / "include",
        root / "libcudacxx" / "include",
    )
    missing = tuple(path.relative_to(root) for path in candidates if not path.is_dir())
    if missing:
        rendered = ", ".join(path.as_posix() for path in missing)
        raise HeaderResolutionError(
            f"{root} looks like a CCCL source checkout but is missing source "
            f"include roots: {rendered}."
        )
    return _unique_existing_dirs(candidates)


def _packaged_header_paths(root: Path) -> tuple[Path, ...]:
    candidates = (
        root,
        root / "include",
    )
    for candidate in candidates:
        if not (candidate / _CUB_PROBE).is_file():
            continue
        if (candidate / "cuda_runtime.h").is_file():
            raise HeaderResolutionError(
                f"{candidate} is a CUDA toolkit include directory, not a "
                "cuda-coop header bundle."
            )
        return _unique_existing_dirs((candidate,))
    return ()


def _configured_cccl_paths(root: Path) -> tuple[Path, ...]:
    resolved_root = root.expanduser().resolve()
    paths = _source_checkout_paths(resolved_root)
    if paths:
        return paths
    paths = _packaged_header_paths(resolved_root)
    if paths:
        return paths
    raise HeaderResolutionError(
        f"{resolved_root} is neither a CCCL source checkout nor a cuda-coop "
        "header bundle."
    )


def _find_source_checkout(start: Path) -> tuple[Path, tuple[Path, ...]] | None:
    current = start.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        paths = _source_checkout_paths(candidate)
        if paths and _belongs_to_source_package(current, candidate):
            return candidate, paths
    return None


def _belongs_to_source_package(start: Path, root: Path) -> bool:
    """Return whether ``start`` belongs to this checkout's source package."""

    package_root = root / "python" / "cuda_coop"
    try:
        start.relative_to(package_root)
    except ValueError:
        return False
    return True


@cache
def _installed_header_root() -> Path:
    resource = files(__package__).joinpath("include")
    return Path(_INSTALLED_HEADER_CONTEXTS.enter_context(as_file(resource)))


def _installed_include_paths() -> CoopIncludePaths:
    cccl = _packaged_header_paths(_installed_header_root())

    if not any((path / _CUB_PROBE).is_file() for path in cccl):
        raise HeaderResolutionError(
            "The installed cuda-coop wheel does not contain its bundled CUB headers."
        )
    return CoopIncludePaths(
        cccl=cccl,
        cuda=_cuda_include_paths(),
        origin="cuda-coop wheel header bundle",
    )


def _select_cuda_include_path(
    paths: Iterable[Path | str | None],
) -> tuple[Path, ...]:
    candidates = _unique_existing_dirs(paths)
    for candidate in candidates:
        if (candidate / _CUDA_PROBE).is_file():
            return (candidate,)
    return ()


def _cuda_include_paths() -> tuple[Path, ...]:
    try:
        from cuda.pathfinder import find_nvidia_header_directory

        cuda_include = find_nvidia_header_directory("cudart")
    except (ImportError, RuntimeError):
        cuda_include = None

    # Pathfinder already applies its own ordered discovery policy.  A valid
    # result is authoritative so headers from another Toolkit cannot be mixed
    # into the same provider compilation.
    discovered = _select_cuda_include_path((cuda_include,))
    if discovered:
        return discovered

    configured = tuple(
        Path(root).expanduser() / "include"
        for env_name in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT")
        if (root := os.environ.get(env_name))
    )
    return _select_cuda_include_path((*configured, Path("/usr/local/cuda/include")))


def _validate_required_headers(
    required_headers: Iterable[str],
    *,
    cccl_paths: tuple[Path, ...],
    origin: str,
) -> None:
    missing: list[str] = []
    for header in dict.fromkeys(required_headers):
        relative = Path(header)
        if relative.is_absolute() or ".." in relative.parts:
            raise HeaderResolutionError(f"Invalid CCCL header path: {header!r}.")
        if not any((include_path / relative).is_file() for include_path in cccl_paths):
            missing.append(header)
    if missing:
        raise HeaderResolutionError(
            f"The selected {origin} is missing required CCCL headers "
            f"({', '.join(missing)}). cuda.coop does not fall back to CUDA "
            "Toolkit CCCL headers."
        )


def resolve_include_paths(
    *,
    start: Path,
    configured_roots: Iterable[Path | str | None] = (),
    required_headers: Iterable[str] = (),
) -> CoopIncludePaths:
    """Resolve source-tree or wheel-bundled CCCL headers, never toolkit CUB."""

    configured = next((Path(root) for root in configured_roots if root), None)
    if configured is not None:
        cccl = _configured_cccl_paths(configured)
        origin = f"configured CCCL root {configured.expanduser().resolve()}"
        paths = CoopIncludePaths(
            cccl=cccl,
            cuda=_cuda_include_paths(),
            origin=origin,
        )
    elif source := _find_source_checkout(start):
        root, cccl = source
        paths = CoopIncludePaths(
            cccl=cccl,
            cuda=_cuda_include_paths(),
            origin=f"CCCL source checkout {root}",
        )
    else:
        paths = _installed_include_paths()

    _validate_required_headers(
        required_headers,
        cccl_paths=paths.cccl,
        origin=paths.origin,
    )
    return paths


__all__ = [
    "CoopIncludePaths",
    "HeaderResolutionError",
    "resolve_include_paths",
]
