# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile-time group-first primitive markers for Numba-CUDA-MLIR."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._thread_group import ThreadGroup

if TYPE_CHECKING:
    from . import ThreadData

_ROOT_SCOPE = __name__.rsplit(".", 1)[0]


def _group_primitive_marker(
    operation: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Mark a group primitive that the whole-function planner must erase."""

    del args, kwargs
    raise RuntimeError(
        f"{_ROOT_SCOPE}.{operation} is a compile-time kernel construct and "
        "must be lowered by the whole-function planner"
    )


def load(
    group: ThreadGroup,
    source: Any,
    output: ThreadData,
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> ThreadData:
    return _group_primitive_marker(
        "load",
        group,
        source,
        output,
        algorithm=algorithm,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
        temp_storage=temp_storage,
    )


def store(
    group: ThreadGroup,
    destination: Any,
    value: Any,
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> None:
    _group_primitive_marker(
        "store",
        group,
        destination,
        value,
        algorithm=algorithm,
        valid_items=valid_items,
        offset=offset,
        temp_storage=temp_storage,
    )


__all__ = [
    "load",
    "store",
]
