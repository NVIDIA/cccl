# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR bindings for ``cuda.coop`` block reduction."""

from __future__ import annotations

from typing import Any, TypeVar

from cuda.coop._core.root_api import _register_qualified_backend
from cuda.coop._core.thread_group import ThreadGroup
from cuda.coop._core.thread_group import this_block as _core_this_block

from ._compiler._activation import _initialize_runtime_hooks

_ScalarT = TypeVar("_ScalarT")


def this_block() -> ThreadGroup:
    """Return the current block's compile-time group descriptor."""

    return _core_this_block()


def reduce(
    group: ThreadGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: Any = None,
    valid_items: Any = None,
    algorithm: Any = None,
) -> _ScalarT:
    """Reduce one scalar per thread and return the block-root result.

    This is a compiler marker. Every thread in the block must participate in
    converged control flow, and only block rank zero may consume the result.
    A runtime ``valid_items`` value must be uniform, positive, and no larger
    than the block size; the provider converts it to CUB's ``int`` parameter.
    """

    del group, value, binary_op, valid_items, algorithm
    raise RuntimeError("cuda.coop.numba_mlir.reduce is a kernel compile-time construct")


def sum(
    group: ThreadGroup,
    value: _ScalarT,
    /,
    *,
    valid_items: Any = None,
    algorithm: Any = None,
) -> _ScalarT:
    """Sum one scalar per thread and return the block-root result.

    This is a compiler marker. Every thread in the block must participate in
    converged control flow, and only block rank zero may consume the result.
    A runtime ``valid_items`` value must be uniform, positive, and no larger
    than the block size; the provider converts it to CUB's ``int`` parameter.
    """

    del group, value, valid_items, algorithm
    raise RuntimeError("cuda.coop.numba_mlir.sum is a kernel compile-time construct")


for _name in ("this_block", "reduce", "sum"):
    globals()[_name].__cuda_coop_backend_member__ = _name
del _name

__all__ = ["ThreadGroup", "this_block", "reduce", "sum"]

_initialize_runtime_hooks()
_register_qualified_backend(__name__)
