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
from functools import lru_cache
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Build-time path helpers for downstream packages that compile against CUDASTF.
#
# Both helpers anchor from `bindings_module.__file__` — the compiled Cython
# extension (.so).  The extension is always a real file on disk (shared
# libraries cannot be zip-imported) and pip always places it in site-packages,
# even for editable installs.  The installed layout is:
#
#   cuda/stf/
#   ├── include/              <- C API + cudax + libcudacxx headers
#   └── <cu12|cu13>/
#       ├── _stf_bindings_impl.so
#       ├── stf_build_config.json   <- compile flags used for this build
#       └── cccl/
#           └── libcccl.c.experimental.stf.so
# ---------------------------------------------------------------------------

# Directory that contains the compiled extension (e.g. .../cuda/stf/cu13/)
_ext_dir = Path(bindings_module.__file__).resolve().parent
# Parent is .../cuda/stf/
_stf_pkg_dir = _ext_dir.parent


@lru_cache()
def get_include_path() -> Path:
    """Return the path to the CUDASTF C API and cudax headers."""
    include_dir = _stf_pkg_dir / "include"
    if include_dir.is_dir():
        return include_dir
    raise RuntimeError(
        "Cannot locate cuda.stf include directory. "
        "The cuda-cccl package may not have been installed correctly."
    )


@lru_cache()
def get_library_path() -> Path:
    """Return the directory containing ``libcccl.c.experimental.stf.so``."""
    lib_dir = _ext_dir / "cccl"
    lib_name = "libcccl.c.experimental.stf.so"
    if (lib_dir / lib_name).exists():
        return lib_dir
    raise RuntimeError(
        f"Cannot locate {lib_name} for CUDA {cuda_version}. "
        "The cuda-cccl package may not have been installed correctly."
    )


@lru_cache()
def get_compile_flags() -> dict:
    """Return the compile flags used to build ``libcccl.c.experimental.stf.so``.

    Downstream packages that compile C++ code against the STF headers should
    use these flags to ensure ABI compatibility.  The returned dict contains::

        {
            "cuda_standard":        20,
            "cxx_standard":         20,
            "cuda_architectures":   "75-real,80-real,...",
            "cuda_toolkit_version": "13.2.35",
            "defines":              ["CCCL_C_EXPERIMENTAL"],
            "nvcc_flags":           ["--expt-relaxed-constexpr", "--extended-lambda"],
        }
    """
    import json

    config_path = _ext_dir / "stf_build_config.json"
    if not config_path.exists():
        raise RuntimeError(
            "Cannot locate stf_build_config.json. "
            "The cuda-cccl package may not have been installed correctly."
        )
    with open(config_path) as f:
        return json.load(f)
