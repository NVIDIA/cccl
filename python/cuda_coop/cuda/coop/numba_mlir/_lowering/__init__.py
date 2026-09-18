# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Primitive-family lowering factories for Numba-CUDA-MLIR.

This package registers exact factory identities for the compiler; it does not
recognize providers by module or function name.
"""

from ._load_store import load as load
from ._load_store import store as store

__all__: tuple[str, ...] = ()
