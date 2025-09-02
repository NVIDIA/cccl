# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from cuda.cccl._cuda_version_utils import get_cuda_path


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
def get_include_paths(probe_file: str = "cub/version.cuh") -> IncludePaths:
    # TODO: once docs env supports Python >= 3.9, we
    # can move this to a module-level import.
    from importlib.resources import as_file, files

    cuda_incl = None
    cuda_path = get_cuda_path()
    if cuda_path is not None:
        cuda_incl = cuda_path / "include"

    with as_file(files("cuda.cccl.headers.include")) as f:
        cccl_incl = Path(f)

    probe_file_path = Path(probe_file)
    if not (cccl_incl / probe_file_path).exists():
        for sp in sys.path:
            cccl_incl = Path(sp).resolve() / "cuda" / "cccl" / "headers" / "include"
            if (cccl_incl / probe_file_path).exists():
                break
        else:
            raise RuntimeError("Unable to locate CCCL include directory.")

    return IncludePaths(
        cuda=cuda_incl,
        libcudacxx=cccl_incl,
        cub=cccl_incl,
        thrust=cccl_incl,
    )
