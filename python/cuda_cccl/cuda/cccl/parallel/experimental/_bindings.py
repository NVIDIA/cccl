# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# _bindings.py is a shim module that imports symbols from a
# _bindings_impl extension module. The shim serves two purposes:
#
# 1. Import a CUDA-specific extension. The cuda.cccl wheel ships with multiple
#   extensions, one for each CUDA version. At runtime, this shim chooses the
#   appropriate extension based on the detected CUDA version, and imports all
#   symbols from it.
#
# 2. Preload `nvrtc` and `nvJitLink` before importing the extension.
#   These shared libraries are indirect dependencies, pulled in via the direct
#   dependency `cccl.c.parallel`. To ensure reliable symbol resolution at
#   runtime, we explicitly load them first using `cuda.pathfinder`.
#   Without this step, importing the Cython extension directly may fail or behave
#   inconsistently depending on environment setup and dynamic linker behavior.
#   This indirection ensures the right loading order, regardless of how
#   `_bindings` is first imported across the codebase.

import importlib

from cuda.cccl._cuda_version_utils import detect_cuda_version, get_recommended_extra
from cuda.pathfinder import (  # type: ignore[import-not-found]
    load_nvidia_dynamic_lib,
)


def _load_cuda_libraries():
    # Load appropriate libraries for the detected CUDA version
    for libname in ("nvrtc", "nvJitLink"):
        load_nvidia_dynamic_lib(libname)


_load_cuda_libraries()


# Import the appropriate bindings implementation depending on what
# CUDA version is available:
cuda_version = detect_cuda_version()
if cuda_version not in [12, 13]:
    raise RuntimeError(
        f"Unsupported CUDA version: {cuda_version}. Only CUDA 12 and 13 are supported."
    )

try:
    extra_name = get_recommended_extra(cuda_version)
    bindings_module = importlib.import_module(
        f".{extra_name}._bindings_impl", __package__
    )
    # Import all symbols from the module
    globals().update(bindings_module.__dict__)
except ImportError as e:
    raise ImportError(
        f"Failed to import CUDA CCCL bindings for CUDA {cuda_version}. "
    ) from e
