# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative load and store entry points.

These frontends validate the shared algorithm subset and delegate one call to
the active compiler backend. ThreadData allocation and backend-specific CUB
selection remain outside this module.
"""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _LOAD_STORE_ALGORITHMS,
    _group_primitive_marker,
    _portable_selector,
)
from ._payload import ThreadDataLike


def load(
    group: ThreadGroup,
    source: Any,
    output: ThreadDataLike[Any],
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> ThreadDataLike[Any]:
    """Load values cooperatively through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "load", "algorithm", algorithm, _LOAD_STORE_ALGORITHMS
    )

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
    """Store values cooperatively through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "store", "algorithm", algorithm, _LOAD_STORE_ALGORITHMS
    )

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


__all__ = ["load", "store"]
