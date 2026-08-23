# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative scan entry points.

All scan spellings share one selector-normalization and backend-delegation
boundary here. Prefix callbacks and backend-specific algorithms remain
qualified-backend concerns unless represented by this portable surface.
"""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _SCAN_ALGORITHMS,
    _SCAN_MODES,
    _backend_module_name,
    _group_primitive_marker,
    _portable_selector,
)


def scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Scan values across a group through the compiler-selected backend.

    ``mode="exclusive"`` returns a fully defined prefix: the default sum uses
    zero when ``initial_value`` is omitted, while every other ``scan_op``
    requires an explicit ``initial_value``. ``mode="inclusive"`` does not
    accept ``initial_value``.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    mode = _portable_selector("scan", "mode", mode, _SCAN_MODES)
    if (
        _backend_module_name() is not None
        and mode == "inclusive"
        and initial_value is not None
    ):
        raise ValueError(
            "cuda.coop.scan initial_value is not supported for inclusive scans"
        )
    algorithm = _portable_selector(
        "scan",
        "algorithm",
        algorithm,
        _SCAN_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "scan",
        group,
        value,
        mode=mode,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


def exclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return an exclusive prefix sum through the compiler-selected backend.

    The first flattened output position is zero.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "exclusive_sum",
        "algorithm",
        algorithm,
        _SCAN_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "exclusive_sum",
        group,
        value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


def inclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return an inclusive prefix sum through the compiler-selected backend.

    Every output position is defined.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "inclusive_sum",
        "algorithm",
        algorithm,
        _SCAN_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "inclusive_sum",
        group,
        value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


def exclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return an exclusive scan through the compiler-selected backend.

    The default sum uses zero when ``initial_value`` is omitted. Every other
    ``scan_op`` requires an explicit ``initial_value`` so that the first
    flattened output position is defined.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "exclusive_scan",
        "algorithm",
        algorithm,
        _SCAN_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "exclusive_scan",
        group,
        value,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


def inclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return an inclusive scan through the compiler-selected backend.

    Every output position is defined; inclusive scans do not accept an initial
    value in the portable profile.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "inclusive_scan",
        "algorithm",
        algorithm,
        _SCAN_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "inclusive_scan",
        group,
        value,
        scan_op=scan_op,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


__all__ = [
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "scan",
]
