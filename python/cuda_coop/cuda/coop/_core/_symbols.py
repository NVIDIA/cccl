# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stable semantic tokens for cooperative provider artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any


def semantic_token(value: Any) -> Any:
    """Return a deterministic, hashable description of ``value``."""

    if isinstance(value, type):
        return "type", value.__module__, value.__qualname__
    if isinstance(value, Enum):
        return type(value).__qualname__, value.value
    if isinstance(value, Mapping):
        tokens = (
            (semantic_token(key), semantic_token(item)) for key, item in value.items()
        )
        return tuple(sorted(tokens, key=repr))
    if isinstance(value, (tuple, list)):
        return tuple(semantic_token(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return "repr", repr(value)
    return value


__all__ = ["semantic_token"]
