# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# _stf_bindings.py is a shim module that imports symbols from a
# _stf_bindings_impl extension module. The shim serves the same purposes as
# cuda.compute._bindings:
#
# 1. Import a CUDA-specific extension. The cuda-stf wheel ships
#    cuda/stf/_experimental/cu12/ and cuda/stf/_experimental/cu13/; at runtime
#    this shim chooses based on the detected CUDA version and imports all
#    symbols from the matching extension.
#
# 2. Preload `nvrtc` and `nvJitLink` before importing the extension (indirect
#    dependencies via cccl.c.experimental.stf).

from __future__ import annotations

import ctypes
import importlib

from cuda.pathfinder import (  # type: ignore[import-not-found]
    load_nvidia_dynamic_lib,
)

_SUPPORTED_CUDA_VERSIONS = {12, 13}

# Deliberate public API of the compiled bindings. Exporting an explicit list
# (rather than every non-underscore name in the extension module) avoids
# leaking implementation imports such as ``np``, ``ctypes``, ``warnings`` and
# ``IntFlag`` into ``cuda.stf._experimental._stf_bindings``.
_BINDING_EXPORTS = (
    "AccessMode",
    "CudaStream",
    "LaunchableGraph",
    "async_resources",
    "cond",
    "context",
    "cuda_kernel",
    "data_place",
    "dep",
    "exec_place",
    "exec_place_grid",
    "exec_place_resources",
    "green_context_helper",
    "green_ctx_view",
    "logical_data",
    "machine_init",
    "read",
    "rw",
    "stackable_context",
    "stackable_logical_data",
    "stackable_task",
    "stf_cai",
    "task",
    "write",
)


def _load_cuda_libraries():
    # The compiled bindings resolve cudart symbols from the process image
    # (they link only the CUDA driver), so cudart must be visible globally
    # even when another package already loaded it RTLD_LOCAL.
    cudart = load_nvidia_dynamic_lib("cudart")
    ctypes.CDLL(cudart.abs_path, mode=ctypes.RTLD_GLOBAL)
    for libname in ("nvrtc", "nvJitLink"):
        load_nvidia_dynamic_lib(libname)


def _select_cuda_extra():
    try:
        from ._cuda_version_utils import (  # noqa: PLC0415
            detect_cuda_version,
            get_recommended_extra,
        )
    except ImportError as e:
        raise ImportError(
            "CUDASTF bindings require cuda-bindings to detect the CUDA version. "
            "Reinstall cuda-stf with a CUDA extra (for example, "
            "`pip install cuda-stf[cu13]`)."
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
    missing = []
    for name in _BINDING_EXPORTS:
        try:
            globals()[name] = getattr(bindings_module, name)
        except AttributeError:
            missing.append(name)
    if missing:
        raise ImportError(
            "CUDASTF bindings extension is missing expected symbols: "
            + ", ".join(sorted(missing))
        )


__all__ = list(_BINDING_EXPORTS)


_load_cuda_libraries()

cuda_version, extra_name = _select_cuda_extra()
module_suffix = f".{extra_name}._stf_bindings_impl"

try:
    bindings_module = importlib.import_module(module_suffix, __package__)
    _export_public_symbols(bindings_module)
except ImportError as e:
    raise ImportError(
        f"CUDASTF bindings for CUDA {cuda_version} are not available: {e}. "
        f"Reinstall cuda-stf with the matching extra (e.g. `pip install cuda-stf[cu{cuda_version}]`)."
    ) from e
