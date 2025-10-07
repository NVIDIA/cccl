# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# _bindings.py is a shim module that imports symbols from a
# _bindings_impl extension module. The shim serves the following purposes:
#
# 1. Import a CUDA-specific extension. The cuda.cccl wheel ships with multiple
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
# 3. On Windows, add the directory containing cccl.c.parallel's dependent DLL
#    (e.g. cuda/cccl/parallel/experimental/cu13/_bindings_impl.cp312-win_amd64.pyd)
#    to the current process's DLL search path using `os.add_dll_directory`.

import importlib
import os

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

# `extra_name` is one of "cu12", "cu13", etc.
extra_name = get_recommended_extra(cuda_version)
module_suffix = f".{extra_name}._bindings_impl"
module_fullname = __package__ + module_suffix

# On Windows, ensure the dependent DLLs next to the extension are discoverable.
# The extension lives at .../experimental/<extra_name>/_bindings_impl.*.pyd
# and its dependent DLLs are under .../experimental/<extra_name>/cccl/.
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

try:
    bindings_module = importlib.import_module(module_suffix, __package__)
    # Import all symbols from the module
    globals().update(bindings_module.__dict__)
except ImportError as e:
    raise ImportError(
        f"Failed to import CUDA CCCL bindings for CUDA {cuda_version}. "
    ) from e
