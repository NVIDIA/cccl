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

from cuda.pathfinder import (  # type: ignore[import-not-found]
    load_nvidia_dynamic_lib,
)

_SUPPORTED_CUDA_VERSIONS = {12, 13}


def _load_cuda_libraries():
    for libname in ("nvrtc", "nvJitLink"):
        load_nvidia_dynamic_lib(libname)


def _select_cuda_extra():
    try:
        from cuda.cccl._cuda_version_utils import (  # noqa: PLC0415
            detect_cuda_version,
            get_recommended_extra,
        )
    except ImportError as e:
        raise ImportError(
            "CUDASTF bindings require cuda-bindings to detect the CUDA version. "
            "Reinstall cuda-cccl with a CUDA extra (for example, "
            "`pip install cuda-cccl[cu13]`)."
        ) from e

    cuda_version = detect_cuda_version()
    if cuda_version is None:
        raise ImportError("Unable to detect CUDA version for CUDASTF bindings.")

    if cuda_version in _SUPPORTED_CUDA_VERSIONS:
        return cuda_version, get_recommended_extra(cuda_version)

    # Future CUDA majors should fail through the normal extension import path
    # until a matching wheel extra is available, not through an early RuntimeError.
    return cuda_version, f"cu{cuda_version}"


def _export_public_symbols(bindings_module):
    for name, value in bindings_module.__dict__.items():
        if not name.startswith("_"):
            globals()[name] = value


_load_cuda_libraries()

cuda_version, extra_name = _select_cuda_extra()
module_suffix = f".{extra_name}._stf_bindings_impl"

try:
    bindings_module = importlib.import_module(module_suffix, __package__)
    _export_public_symbols(bindings_module)
except ImportError as e:
    raise ImportError(
        f"CUDASTF bindings for CUDA {cuda_version} are not available: {e}. "
        f"Reinstall cuda-cccl with the matching extra (e.g. `pip install cuda-cccl[cu{cuda_version}]`)."
    ) from e
