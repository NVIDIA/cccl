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

bindings_module = importlib.import_module(module_suffix, __package__)
globals().update(bindings_module.__dict__)

# Build-time path helpers (for downstream packages that compile against the STF C API).
# Defined after update() so the extension cannot overwrite them.
from functools import lru_cache
from pathlib import Path
import sys as _sys


@lru_cache()
def get_include_path():
    """Return the path to CUDASTF C API and cudax headers. See cuda.stf.get_include_path."""
    pkg_dir = Path(__file__).resolve().parent
    include_dir = pkg_dir / "include"
    if include_dir.is_dir():
        return include_dir
    for sp in _sys.path:
        candidate = Path(sp) / "cuda" / "stf" / "include"
        if candidate.is_dir():
            return candidate
    raise RuntimeError(
        "Cannot locate cuda.stf include directory. Run the cuda-cccl build first "
        "(pip install -e . or build the wheel); the build copies headers into cuda/stf/include."
    )


@lru_cache()
def get_library_path():
    """Return the path to the directory containing libcccl.c.experimental.stf.so."""
    lib_name = "libcccl.c.experimental.stf.so"
    # Prefer directory next to the loaded extension (works for installed and editable)
    ext_dir = Path(bindings_module.__file__).resolve().parent
    lib_dir = ext_dir / "cccl"
    if (lib_dir / lib_name).exists():
        return lib_dir
    pkg_dir = Path(__file__).resolve().parent
    lib_dir = pkg_dir / extra_name / "cccl"
    if (lib_dir / lib_name).exists():
        return lib_dir
    for sp in _sys.path:
        candidate = Path(sp) / "cuda" / "stf" / extra_name / "cccl"
        if (candidate / lib_name).exists():
            return candidate
    raise RuntimeError(
        f"Cannot locate {lib_name} for CUDA {cuda_version}. "
        "The cuda-cccl package may not have been installed correctly."
    )
