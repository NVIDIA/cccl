# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stateful device-callable descriptor used by callback-bearing primitives."""

from typing import Generic

from typing_extensions import TypeVar

_OpT = TypeVar("_OpT")

class StatefulFunction(Generic[_OpT]):
    """Device callable paired with explicit state for generated wrappers."""

    op: _OpT
    dtype: object
    name: str | None

    def __init__(
        self,
        op: _OpT,
        dtype: object,
        name: str | None = None,
    ) -> None:
        """Pair ``op`` with its state ``dtype`` and optional generated name."""

__all__ = ["StatefulFunction"]
