# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR bindings for ``cuda.coop`` block reduction."""

from __future__ import annotations

from cuda.coop._core.api._dispatch import _register_qualified_backend

from ._compiler._activation import _initialize_runtime_hooks
from ._group_reduce import reduce, sum
from ._thread_group import ThreadGroup, this_block

__all__ = ["ThreadGroup", "this_block", "reduce", "sum"]

_initialize_runtime_hooks()
_register_qualified_backend(__name__)
