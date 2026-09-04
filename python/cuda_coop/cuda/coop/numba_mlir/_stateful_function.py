# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe a device callable paired with explicit runtime state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._semantic import _normalize_numba_callable


@dataclass(frozen=True, slots=True)
class StatefulFunction:
    """Device callable whose first argument is a pointer to explicit state."""

    op: Any
    dtype: Any
    name: str | None = None

    def __post_init__(self) -> None:
        normalized = _normalize_numba_callable(self.op)
        if not callable(normalized):
            raise TypeError("StatefulFunction op must be callable")
        if self.dtype is None:
            raise TypeError("StatefulFunction dtype must be provided")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError("StatefulFunction name must be a non-empty string")
        object.__setattr__(self, "op", normalized)


__all__ = ["StatefulFunction"]
