# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import-light probes for public CUTLASS values used by the Prims path."""

from __future__ import annotations

import sys
from typing import Any


def is_cutlass_array_operand(value: Any, *, method: str | None = None) -> bool:
    """Return whether *value* is a public ``cutlass.Array`` operand.

    This local probe never imports CUTLASS itself. It resolves the public class
    only when another activation path, including default root auto-registration,
    has already loaded the CUTLASS runtime.
    """
    cutlass_module = sys.modules.get("cutlass")
    array_type = (
        getattr(cutlass_module, "Array", None) if cutlass_module is not None else None
    )
    if not isinstance(array_type, type) or not isinstance(value, array_type):
        return False

    if method is not None:
        return callable(getattr(value, method, None))

    return callable(getattr(value, "load", None)) or callable(
        getattr(value, "store", None)
    )


__all__ = ["is_cutlass_array_operand"]
