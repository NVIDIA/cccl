# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Preload `nvrtc` and `nvJitLink` before importing the Cython extension.
# These shared libraries are indirect dependencies, pulled in via the direct
# dependency `cccl.c.parallel`. To ensure reliable symbol resolution at
# runtime, we explicitly load them first using `cuda.bindings.path_finder`.
#
# Without this step, importing the Cython extension directly may fail or behave
# inconsistently depending on environment setup and dynamic linker behavior.
# This indirection ensures the right loading order, regardless of how
# `_bindings` is first imported across the codebase.
#
# See also:
# https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/cuda/bindings/_path_finder/README.md

from cuda.bindings.path_finder import (  # type: ignore[import-not-found]
    _load_nvidia_dynamic_library,
)

for libname in ("nvrtc", "nvJitLink"):
    _load_nvidia_dynamic_library(libname)

from ._bindings_impl import *  # noqa: E402 F403
