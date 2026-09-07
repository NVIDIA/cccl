# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe a device callable paired with explicit compiler-visible state."""

from ._semantic import _normalize_numba_callable


class StatefulFunction:
    """Stateful Python device callable used by cooperative primitives."""

    def __init__(self, op, dtype, name=None):
        self.op = _normalize_numba_callable(op)
        self.dtype = dtype
        self.name = name


__all__ = ["StatefulFunction"]
