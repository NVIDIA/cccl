# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared diagnostic for group primitives that escape compiler lowering."""

from __future__ import annotations

from typing import Any


def group_primitive_marker(operation: str, *args: Any, **kwargs: Any) -> Any:
    """Reject execution of a marker that must be erased during compilation."""

    del args, kwargs
    raise RuntimeError(
        f"cuda.coop.numba_mlir.{operation} is a kernel compile-time construct "
        "and must be lowered by the whole-function planner"
    )


__all__ = ["group_primitive_marker"]
