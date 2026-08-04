# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUDA version detection utilities for :mod:`cuda.compute`."""

from typing import Optional

try:
    import cuda.bindings
except ModuleNotFoundError as error:
    if error.name != "cuda.bindings":
        raise
    raise ImportError(
        "cuda.bindings is not installed. Please install the appropriate extra "
        "cuda-compute[cu12] or cuda-compute[cu13]."
    ) from None


def detect_cuda_version() -> Optional[int]:
    cuda_version = cuda.bindings.__version__
    return int(cuda_version.split(".")[0])


def get_recommended_extra(cuda_version: Optional[int]) -> str:
    """Get the recommended pip extra for the detected CUDA version."""
    if cuda_version == 13:
        return "cu13"
    else:
        return "cu12"
