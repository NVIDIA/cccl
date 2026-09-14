# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib
import importlib.util
import os

from cuda.cccl._cuda_version_utils import detect_cuda_version, get_recommended_extra

try:
    from ._build_info import USING_V2  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(
        "cuda.compute._device_copy_impl is available only from installed "
        "cuda.compute v2 builds"
    ) from e

if not USING_V2:
    raise ImportError(
        "cuda.compute._device_copy_impl is available only in v2 HostJIT builds"
    )

# Reuse the existing bindings shim for CUDA library preloading and, on Windows,
# for registering the versioned extension directory with the DLL loader.
from . import _bindings as _bindings  # noqa: F401

cuda_version = detect_cuda_version()
if cuda_version not in [12, 13]:
    raise RuntimeError(
        f"Unsupported CUDA version: {cuda_version}. Only CUDA 12 and 13 are supported."
    )

extra_name = get_recommended_extra(cuda_version)
module_suffix = f".{extra_name}._device_copy_impl"
module_fullname = __package__ + module_suffix

if os.name == "nt":
    spec = importlib.util.find_spec(module_fullname)
    if spec and spec.origin:
        dll_dir = os.path.join(os.path.dirname(spec.origin), "cccl")
        if os.path.isdir(dll_dir):
            try:
                add_dll_directory = getattr(os, "add_dll_directory", None)
                if add_dll_directory is not None:
                    _cccl_dll_dir_handle = add_dll_directory(dll_dir)  # noqa: F841
            except Exception:
                pass

bindings_module = importlib.import_module(module_suffix, __package__)
globals().update(bindings_module.__dict__)
