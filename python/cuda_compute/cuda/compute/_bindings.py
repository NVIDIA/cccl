# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# _bindings.py is a shim module that imports symbols from a
# _bindings_impl extension module. The shim serves the following purposes:
#
# 1. Import a CUDA-specific extension. The cuda-compute wheel ships with multiple
#    extensions, one for each CUDA version. At runtime, this shim chooses the
#    appropriate extension based on the detected CUDA version, and imports all
#    symbols from it.
#
# 2. Preload `nvrtc` and `nvJitLink` before importing the extension.
#    These shared libraries are indirect dependencies, pulled in via the direct
#    dependency `cccl.c.parallel`. To ensure reliable symbol resolution at
#    runtime, we explicitly load them first using `cuda.pathfinder`.
#    Without this step, importing the Cython extension directly may fail or behave
#    inconsistently depending on environment setup and dynamic linker behavior.
#    This indirection ensures the right loading order, regardless of how
#    `_bindings` is first imported across the codebase.
#
# 3. On Windows, add the directory containing cccl.c.parallel's dependent DLLs
#    (e.g. cuda/compute/cu13/cccl) to the current process's DLL search path
#    using `os.add_dll_directory`.

from __future__ import annotations

import importlib
import os

from cuda.pathfinder import (  # type: ignore[import-not-found]
    load_nvidia_dynamic_lib,
)

from ._cuda_version_utils import detect_cuda_version, get_recommended_extra


def _load_cuda_libraries():
    # Load appropriate libraries for the detected CUDA version
    libraries = ["nvrtc", "nvJitLink"]
    try:
        from ._build_info import USING_V2  # type: ignore[import-not-found]
    except ImportError:
        USING_V2 = False
    if USING_V2 and os.name == "nt":
        # PR #9583's libnvcc.dll links nvfatbin dynamically on Windows. Load
        # the pip- or system-toolkit copy before importing _bindings_impl so
        # Windows can resolve libnvcc's transitive dependency.
        libraries.append("nvfatbin")

    for libname in libraries:
        load_nvidia_dynamic_lib(libname)


_load_cuda_libraries()


# Import the appropriate bindings implementation depending on what
# CUDA version is available:
cuda_version = detect_cuda_version()
if cuda_version not in [12, 13]:
    raise RuntimeError(
        f"Unsupported CUDA version: {cuda_version}. Only CUDA 12 and 13 are supported."
    )

# `extra_name` is one of "cu12", "cu13", etc.
extra_name = get_recommended_extra(cuda_version)
module_suffix = f".{extra_name}._bindings_impl"
module_fullname = __package__ + module_suffix

# On Windows, ensure the dependent DLLs next to the extension are discoverable.
# The extension lives at .../compute/<extra_name>/_bindings_impl.*.pyd and its
# dependent DLLs are under .../compute/<extra_name>/cccl/.
if os.name == "nt":
    spec = importlib.util.find_spec(module_fullname)
    if spec and spec.origin:
        dll_dir = os.path.join(os.path.dirname(spec.origin), "cccl")
        if os.path.isdir(dll_dir):
            # Assign the DLL directory handle to a global such that it stays
            # alive for the lifetime of this module (and thus, keeps the DLL
            # directory in the search path).
            try:
                _cccl_dll_dir_handle = os.add_dll_directory(dll_dir)  # noqa: F841
            except Exception:
                pass

_BINDINGS_AVAILABLE = False

try:
    bindings_module = importlib.import_module(module_suffix, __package__)
    # Import all symbols from the module
    globals().update(bindings_module.__dict__)
    _BINDINGS_AVAILABLE = True
except ImportError as e:
    import warnings

    warnings.warn(
        f"cuda.compute bindings for CUDA {cuda_version} not available: {e}",
        RuntimeWarning,
    )
