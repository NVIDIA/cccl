# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from cuda.bindings.path_finder import (  # type: ignore[import-not-found]
    _load_nvidia_dynamic_library,
)

for libname in ("nvrtc", "nvJitLink"):
    logging.info(str(_load_nvidia_dynamic_library(libname)))
