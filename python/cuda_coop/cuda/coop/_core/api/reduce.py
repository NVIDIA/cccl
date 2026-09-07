# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative reduction entry points.

The root functions retain the conservative common reduction controls and
delegate to the active backend. Semantic planning and CUDAX/CUB selection live
in the portable group family and backend lowering layers.
"""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import _REDUCE_ALGORITHMS, _group_primitive_marker, _portable_selector


def reduce(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    binary_op: Any = None,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Reduce values across a group through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "reduce",
        "algorithm",
        algorithm,
        _REDUCE_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "reduce",
        group,
        value,
        binary_op=binary_op,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


def sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Sum values across a group through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "sum",
        "algorithm",
        algorithm,
        _REDUCE_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "sum",
        group,
        value,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


__all__ = ["reduce", "sum"]
