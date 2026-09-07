# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative CUDA primitives for Python kernel DSLs."""

from __future__ import annotations

import importlib.metadata

from ._core._auto_registration import _auto_register_cutlass
from ._core.root_api import ThreadData, ThreadGroup, load, store, this_block


def _package_version() -> str:
    try:
        return importlib.metadata.version("cuda-coop")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


__version__ = _package_version()

__all__ = [
    "__version__",
    "ThreadData",
    "ThreadGroup",
    "this_block",
    "load",
    "store",
]

_auto_register_cutlass()
del _auto_register_cutlass
