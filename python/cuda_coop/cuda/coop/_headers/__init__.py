# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve the private CCCL headers used by cooperative providers."""

from __future__ import annotations

import atexit
import functools
import os
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path

_CUB_PROBE = Path("cub/version.cuh")
_CUDA_PROBE = Path("cuda_runtime.h")
_INSTALLED_HEADER_CONTEXTS = ExitStack()
atexit.register(_INSTALLED_HEADER_CONTEXTS.close)


class HeaderResolutionError(RuntimeError):
    """The package cannot resolve a coherent CCCL header set."""


@dataclass(frozen=True)
class CoopIncludePaths:
    """Ordered CCCL and CUDA include paths with their provenance."""

    cccl: tuple[Path, ...]
    cuda: tuple[Path, ...]
    origin: str

    def as_tuple(self) -> tuple[Path, ...]:
        if not self.cuda:
            raise HeaderResolutionError(
                "Unable to locate a CUDA include directory containing "
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
    return _unique_existing_dirs(
        (
            root / "thrust",
            root / "cub",
            root / "libcudacxx" / "include",
        )
    )


def _packaged_header_paths(root: Path) -> tuple[Path, ...]:
    for candidate in (root, root / "include"):
        if not (candidate / _CUB_PROBE).is_file():
            continue
        if (candidate / _CUDA_PROBE).is_file():
            raise HeaderResolutionError(
                f"{candidate} is a CUDA Toolkit include directory, not a "
                "cuda-coop header bundle"
            )
        return _unique_existing_dirs((candidate,))
    return ()


def _find_source_checkout(start: Path) -> tuple[Path, tuple[Path, ...]] | None:
    current = start.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        source_paths = _source_checkout_paths(candidate)
        if source_paths:
            return candidate, source_paths
    return None


@functools.lru_cache(maxsize=1)
def _installed_header_root() -> Path:
    resource = files(__package__).joinpath("include")
    return Path(_INSTALLED_HEADER_CONTEXTS.enter_context(as_file(resource)))


def _select_cuda_include_path(
    paths: Iterable[Path | str | None],
) -> tuple[Path, ...]:
    for candidate in _unique_existing_dirs(paths):
        if (candidate / _CUDA_PROBE).is_file():
            return (candidate,)
    return ()


def _cuda_include_paths() -> tuple[Path, ...]:
    try:
        from cuda.pathfinder import find_nvidia_header_directory

        cuda_include = find_nvidia_header_directory("cudart")
    except (ImportError, RuntimeError):
        cuda_include = None
    discovered = _select_cuda_include_path((cuda_include,))
    if discovered:
        return discovered
    configured = tuple(
        Path(root).expanduser() / "include"
        for name in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT")
        if (root := os.environ.get(name))
    )
    return _select_cuda_include_path((*configured, Path("/usr/local/cuda/include")))


def _installed_include_paths() -> CoopIncludePaths:
    cccl_paths = _packaged_header_paths(_installed_header_root())
    if not any((path / _CUB_PROBE).is_file() for path in cccl_paths):
        raise HeaderResolutionError(
            "The installed cuda-coop wheel does not contain its private CUB headers"
        )
    return CoopIncludePaths(
        cccl=cccl_paths,
        cuda=_cuda_include_paths(),
        origin="cuda-coop wheel header bundle",
    )


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
            raise HeaderResolutionError(f"Invalid CCCL header path: {header!r}")
        if not any((include_path / relative).is_file() for include_path in cccl_paths):
            missing.append(header)
    if missing:
        raise HeaderResolutionError(
            f"The selected {origin} is missing required CCCL headers "
            f"({', '.join(missing)}). cuda.coop does not fall back to CUDA "
            "Toolkit CUB headers"
        )


def resolve_include_paths(
    *,
    start: Path,
    required_headers: Iterable[str] = (),
) -> CoopIncludePaths:
    """Resolve source-tree or wheel-bundled CCCL headers, never Toolkit CUB."""

    if source := _find_source_checkout(start):
        root, cccl_paths = source
        paths = CoopIncludePaths(
            cccl=cccl_paths,
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


__all__ = ["CoopIncludePaths", "HeaderResolutionError", "resolve_include_paths"]
