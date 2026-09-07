# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared block-primitive semantic vocabulary."""

from __future__ import annotations

from enum import Enum
from numbers import Integral
from typing import Any


def normalize_positive_int(name: str, value: Any) -> int:
    """Return a positive integral option without accepting boolean values."""

    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def normalize_block_dim(value: Any) -> tuple[int, int, int]:
    """Normalize an explicit three-dimensional positive block shape."""

    try:
        dimensions = tuple(value)
    except TypeError as exc:
        raise ValueError("block_dim must contain three positive dimensions") from exc
    if len(dimensions) != 3:
        raise ValueError("block_dim must contain three positive dimensions")
    try:
        x, y, z = (
            normalize_positive_int("block_dim", dimension) for dimension in dimensions
        )
    except ValueError as exc:
        raise ValueError("block_dim must contain three positive dimensions") from exc
    return x, y, z


def normalize_boolean_option(name: str, value: Any) -> bool:
    """Normalize native and NumPy boolean scalars without importing NumPy."""

    if isinstance(value, bool):
        return value
    for value_type in type(value).__mro__:
        module_root = (value_type.__module__ or "").split(".", 1)[0]
        if module_root == "numpy" and value_type.__name__ in {"bool", "bool_"}:
            return bool(value)
    raise ValueError(f"{name} must be a boolean")


class BlockTileBoundary(str, Enum):
    """External neighbor items supplied for a cooperative block tile."""

    NONE = "none"
    PREDECESSOR = "predecessor"
    SUCCESSOR = "successor"
    BOTH = "both"

    @classmethod
    def from_presence(
        cls,
        *,
        predecessor: bool,
        successor: bool,
    ) -> "BlockTileBoundary":
        if predecessor and successor:
            return cls.BOTH
        if predecessor:
            return cls.PREDECESSOR
        if successor:
            return cls.SUCCESSOR
        return cls.NONE

    @property
    def has_predecessor(self) -> bool:
        return self in {
            BlockTileBoundary.PREDECESSOR,
            BlockTileBoundary.BOTH,
        }

    @property
    def has_successor(self) -> bool:
        return self in {
            BlockTileBoundary.SUCCESSOR,
            BlockTileBoundary.BOTH,
        }


__all__ = [
    "BlockTileBoundary",
    "normalize_block_dim",
    "normalize_boolean_option",
    "normalize_positive_int",
]
