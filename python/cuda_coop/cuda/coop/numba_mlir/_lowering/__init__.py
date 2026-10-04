# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Hold the private Load/Store factories and the Numba core adapter.

Group planning selects a factory in ``_load_store`` after validating a
public call. The package re-exports the storage-free block ``load`` and
``store``. Calling a factory specializes device code; it does not move data.
"""

from ._load_store import load as load
from ._load_store import store as store

__all__: tuple[str, ...] = ()
