# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
CUDA version detection utilities shared across the cccl package.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

import cuda.bindings


def detect_cuda_version() -> Optional[int]:
    cuda_version = cuda.bindings.__version__
    return int(cuda_version.split(".")[0])


def get_cuda_path() -> Optional[Path]:
    """Get the CUDA installation path."""
    cuda_path_str = os.environ.get("CUDA_PATH")
    if cuda_path_str:
        cuda_path = Path(cuda_path_str)
        if cuda_path.exists():
            return cuda_path

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        return Path(nvcc_path).parent.parent

    default_path = Path("/usr/local/cuda")
    if default_path.exists():
        return default_path

    return None


def get_recommended_extra(cuda_version: Optional[int]) -> str:
    """Get the recommended pip extra for the detected CUDA version."""
    if cuda_version == 13:
        return "cu13"
    else:
        return "cu12"
