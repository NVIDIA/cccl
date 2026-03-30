# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# _stf_bindings.py is a shim module that imports symbols from a
# _stf_bindings_impl extension module. The shim serves the same purposes as
# cuda.compute._bindings:
#
# 1. Import a CUDA-specific extension. The wheel ships cuda/stf/cu12/ and
#   cuda/stf/cu13/; at runtime this shim chooses based on the detected CUDA
#   version and imports all symbols from the matching extension.
#
# 2. Preload `nvrtc` and `nvJitLink` before importing the extension (indirect
#   dependencies via cccl.c.parallel / cccl.c.experimental.stf).

from __future__ import annotations

import importlib

from cuda.cccl._cuda_version_utils import detect_cuda_version, get_recommended_extra
from cuda.pathfinder import (  # type: ignore[import-not-found]
    load_nvidia_dynamic_lib,
)


def _load_cuda_libraries():
    """
    Preload CUDA libraries to ensure proper symbol resolution.

    These libraries are indirect dependencies pulled in via cccl.c.parallel.
    Preloading ensures reliable symbol resolution regardless of dynamic linker behavior.
    """
    import warnings

    for libname in ("nvrtc", "nvJitLink"):
        try:
            load_nvidia_dynamic_lib(libname)
        except Exception as e:
            # Log warning but don't fail - the extension might still work
            # if the libraries are already loaded or available through other means
            warnings.warn(
                f"Failed to preload CUDA library '{libname}': {e}. "
                f"STF bindings may fail to load if {libname} is not available.",
                RuntimeWarning,
                stacklevel=2,
            )


_load_cuda_libraries()


# Import the appropriate bindings implementation depending on what
# CUDA version is available:
cuda_version = detect_cuda_version()
if cuda_version not in [12, 13]:
    raise RuntimeError(
        f"Unsupported CUDA version: {cuda_version}. Only CUDA 12 and 13 are supported."
    )

extra_name = get_recommended_extra(cuda_version)
module_suffix = f".{extra_name}._stf_bindings_impl"

_BINDINGS_AVAILABLE = False

try:
    bindings_module = importlib.import_module(module_suffix, __package__)
    # Import all symbols from the module
    globals().update(bindings_module.__dict__)
    _BINDINGS_AVAILABLE = True
except ImportError as e:
    import warnings

    warnings.warn(
        f"CUDASTF bindings for CUDA {cuda_version} not available: {e}",
        RuntimeWarning,
    )
