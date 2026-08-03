# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path

from cuda.pathfinder import (
    find_nvidia_header_directory,  # type: ignore[import-not-found]
)


@dataclass
class IncludePaths:
    """CUDA Toolkit and CCCL include directories in compiler search order."""

    cuda: Path | None
    libcudacxx: Path | None
    cub: Path | None
    thrust: Path | None

    def as_tuple(
        self,
    ) -> tuple[Path | None, Path | None, Path | None, Path | None]:
        """Return higher-level through lower-level include directories."""

        return (self.thrust, self.cub, self.libcudacxx, self.cuda)


def _find_bundled_include_directory(probe_file: Path) -> Path:
    with as_file(files("cuda.cccl.headers.include")) as resource_path:
        include_directory = Path(resource_path)

    if (include_directory / probe_file).exists():
        return include_directory

    for search_path in sys.path:
        candidate = (
            Path(search_path).resolve() / "cuda" / "cccl" / "headers" / "include"
        )
        if (candidate / probe_file).exists():
            return candidate

    raise RuntimeError("Unable to locate CCCL include directory.")


@lru_cache
def get_include_paths(probe_file: str = "cub/version.cuh") -> IncludePaths:
    """Locate the CUDA Toolkit headers and the headers bundled in this wheel."""

    cuda_include = find_nvidia_header_directory("cudart")
    if cuda_include is None:
        raise RuntimeError("Unable to locate CUDA include directory.")

    cccl_include = _find_bundled_include_directory(Path(probe_file))
    return IncludePaths(
        cuda=Path(cuda_include),
        libcudacxx=cccl_include,
        cub=cccl_include,
        thrust=cccl_include,
    )
