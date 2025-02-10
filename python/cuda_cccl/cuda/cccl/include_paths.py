# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _get_cuda_path() -> Optional[Path]:
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_path = Path(cuda_path)
        if cuda_path.exists():
            return cuda_path

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        return Path(nvcc_path).parent.parent

    default_path = Path("/usr/local/cuda")
    if default_path.exists():
        return default_path

    return None


@dataclass
class IncludePaths:
    cuda: Optional[Path]
    libcudacxx: Optional[Path]
    cub: Optional[Path]
    thrust: Optional[Path]

    def as_tuple(self):
        # Note: higher-level ... lower-level order:
        return (self.thrust, self.cub, self.libcudacxx, self.cuda)


@lru_cache()
def get_include_paths() -> IncludePaths:
    # TODO: once docs env supports Python >= 3.9, we
    # can move this to a module-level import.
    from importlib.resources import as_file, files

    cuda_incl = None
    cuda_path = _get_cuda_path()
    if cuda_path is not None:
        cuda_incl = cuda_path / "include"

    with as_file(files("cuda.cccl.include")) as f:
        cccl_incl = Path(f)

    def cub_version_exists(p):
        return Path(p / "cub" / "version.cuh").exists()

    # if version is not found, find it in sys.path directories
    if not cub_version_exists(cccl_incl):
        cub_ver_fn = "cuda/cccl/include/cub/version.cuh"
        for dir_ in sys.path:
            gl = Path(dir_).glob(cub_ver_fn)
            candidates = list(gl)
            for f in candidates:
                # found cub/version.cuh
                # use one-level-up parent
                cccl_incl = f.parents[1]
                break
    assert cub_version_exists(cccl_incl)

    return IncludePaths(
        cuda=cuda_incl,
        libcudacxx=cccl_incl,
        cub=cccl_incl,
        thrust=cccl_incl,
    )
