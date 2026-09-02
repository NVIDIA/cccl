# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact lowering factories used only by the Numba-CUDA-MLIR rewrite."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .._compiler._operations import register_factory
from ._reduce import block_reduce_builtin, sum


def _register(*factories: Callable[..., Any]) -> None:
    for factory in factories:
        register_factory(factory, operation=factory.__name__, namespace="block")


_register(block_reduce_builtin, sum)

__all__: tuple[str, ...] = ()
