# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Construct fixed-size thread data and expose compiler memory namespaces."""

from __future__ import annotations

import operator
import struct

from ._compiler._activation import _require_runtime

# Annotations keep the runtime namespaces lazy while documenting module
# ownership for introspection and static analysis.
local: object
shared: object


def ThreadData(
    items_per_thread,
    dtype=None,
    *,
    alignas=8,
):
    """Create fixed-size thread-local storage for cooperative operations."""

    if isinstance(items_per_thread, bool):
        raise TypeError("items_per_thread must be an integer")
    try:
        items_per_thread = operator.index(items_per_thread)
    except TypeError as exc:
        raise TypeError("items_per_thread must be an integer") from exc
    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be a positive integer")

    if isinstance(alignas, bool):
        raise TypeError("alignment must be an integer")
    try:
        alignas = operator.index(alignas)
    except TypeError as exc:
        raise TypeError("alignment must be an integer") from exc
    if alignas <= 0:
        raise ValueError("alignment must be a positive integer")
    if alignas & (alignas - 1):
        raise ValueError("alignment must be a power of 2")
    pointer_size = struct.calcsize("P")
    if alignas % pointer_size:
        raise ValueError(f"alignment must be a multiple of {pointer_size}")

    return _require_runtime().local.array(
        items_per_thread,
        dtype,
        alignment=alignas,
    )


def __getattr__(name: str):
    if name in {"local", "shared"}:
        value = getattr(_require_runtime(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = ["ThreadData", "local", "shared"]
