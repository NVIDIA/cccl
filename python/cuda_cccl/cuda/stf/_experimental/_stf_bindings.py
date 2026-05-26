# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# _stf_bindings.py is a shim module that imports symbols from a
# _stf_bindings_impl extension module. The shim serves the same purposes as
# cuda.compute._bindings:
#
# 1. Import a CUDA-specific extension. The cuda-cccl wheel ships
#    cuda/stf/_experimental/cu12/ and cuda/stf/_experimental/cu13/; at runtime
#    this shim chooses based on the detected CUDA version and imports all
#    symbols from the matching extension.
#
# 2. Preload `nvrtc` and `nvJitLink` before importing the extension (indirect
#    dependencies via cccl.c.experimental.stf).

from __future__ import annotations

import importlib

from cuda.cccl._cuda_version_utils import detect_cuda_version, get_recommended_extra
from cuda.pathfinder import (  # type: ignore[import-not-found]
    load_nvidia_dynamic_lib,
)


def _load_cuda_libraries():
    for libname in ("nvrtc", "nvJitLink"):
        load_nvidia_dynamic_lib(libname)


_load_cuda_libraries()

cuda_version = detect_cuda_version()
if cuda_version not in [12, 13]:
    raise RuntimeError(
        f"Unsupported CUDA version: {cuda_version}. Only CUDA 12 and 13 are supported."
    )

extra_name = get_recommended_extra(cuda_version)
module_suffix = f".{extra_name}._stf_bindings_impl"

try:
    bindings_module = importlib.import_module(module_suffix, __package__)
    globals().update(bindings_module.__dict__)
except ImportError as e:
    raise ImportError(
        f"CUDASTF bindings for CUDA {cuda_version} are not available: {e}. "
        f"Reinstall cuda-cccl with the matching extra (e.g. `pip install cuda-cccl[cu{cuda_version}]`)."
    ) from e
