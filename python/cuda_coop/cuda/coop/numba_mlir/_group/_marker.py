# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Report cooperative markers that reach ordinary Python execution.

Kernel compilation replaces these calls with planned device operations.
Reaching this function means that replacement has not occurred.
"""

from __future__ import annotations

from typing import Any


def group_primitive_marker(operation: str, *args: Any, **kwargs: Any) -> Any:
    """Reject execution of a marker that should have been compiler-lowered."""

    del args, kwargs
    raise RuntimeError(
        f"cuda.coop.numba_mlir.{operation} is a compile-time kernel construct "
        "and must be lowered by the whole-function planner"
    )


__all__ = ["group_primitive_marker"]
