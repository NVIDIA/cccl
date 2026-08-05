# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from cuda.pathfinder import (
    find_nvidia_header_directory,  # type: ignore[import-not-found]
)


@dataclass(frozen=True)
class _IncludePaths:
    cuda: Path
    libcudacxx: Path
    cub: Path
    thrust: Path

    def as_tuple(self) -> tuple[Path, Path, Path, Path]:
        # Keep higher-level libraries before their dependencies.
        return (self.thrust, self.cub, self.libcudacxx, self.cuda)


def _private_wheel_include(package_dir: Path) -> Path | None:
    include_dir = package_dir / "_cccl" / "include"
    probes = (
        include_dir / "cub" / "version.cuh",
        include_dir / "thrust" / "version.h",
        include_dir / "cuda" / "std" / "version",
    )
    return include_dir if all(probe.is_file() for probe in probes) else None


def _editable_include_paths(module_file: Path) -> tuple[Path, Path, Path] | None:
    for repository_root in module_file.resolve().parents:
        project_file = repository_root / "python" / "cuda_cccl" / "pyproject.toml"
        libcudacxx = repository_root / "libcudacxx" / "include"
        cub = repository_root / "cub"
        thrust = repository_root / "thrust"
        probes = (
            project_file,
            libcudacxx / "cuda" / "std" / "version",
            cub / "cub" / "version.cuh",
            thrust / "thrust" / "version.h",
        )
        if all(probe.is_file() for probe in probes):
            return libcudacxx, cub, thrust
    return None


@lru_cache(maxsize=1)
def get_include_paths() -> _IncludePaths:
    cuda_include = find_nvidia_header_directory("cudart")
    if cuda_include is None:
        raise RuntimeError("Unable to locate the CUDA include directory.")

    module_file = Path(__file__).resolve()
    private_include = _private_wheel_include(module_file.parent)
    if private_include is not None:
        return _IncludePaths(
            cuda=Path(cuda_include),
            libcudacxx=private_include,
            cub=private_include,
            thrust=private_include,
        )

    try:
        from ._build_info import BUILD_STATE  # type: ignore[import-not-found]
    except ImportError:
        BUILD_STATE = "source"
    if BUILD_STATE == "wheel":
        raise RuntimeError(
            "The cuda-compute wheel is missing its private CCCL header payload. "
            "Reinstall cuda-compute."
        )

    editable_paths = _editable_include_paths(module_file)
    if editable_paths is None:
        raise RuntimeError(
            "Unable to locate cuda-compute's private CCCL headers. Reinstall the "
            "cuda-compute wheel, or use an editable install from a complete CCCL "
            "checkout."
        )

    libcudacxx, cub, thrust = editable_paths
    return _IncludePaths(
        cuda=Path(cuda_include),
        libcudacxx=libcudacxx,
        cub=cub,
        thrust=thrust,
    )
