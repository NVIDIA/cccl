# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact launch facts supplied by the active CUTLASS compiler."""

from __future__ import annotations

import operator
from typing import Any

from ..._core.launch import LaunchFactOrigin, LaunchFacts
from ._runtime import validate_cutlass_runtime


def normalize_block_dim(value: Any) -> tuple[int, int, int] | None:
    """Normalize a static shape without coercing floats or Boolean dimensions."""

    dimensions = value if isinstance(value, (tuple, list)) else (value,)
    if not 1 <= len(dimensions) <= 3:
        return None
    normalized = []
    for dimension in dimensions:
        if isinstance(dimension, bool):
            return None
        try:
            dimension = operator.index(dimension)
        except TypeError:
            return None
        if dimension <= 0:
            return None
        normalized.append(dimension)
    normalized.extend([1] * (3 - len(normalized)))
    return tuple(normalized)


def block_dim_product(value: Any) -> int | None:
    dimensions = normalize_block_dim(value)
    if dimensions is None:
        return None
    x, y, z = dimensions
    return x * y * z


def launch_facts_from_cutlass_api(
    facts: Any,
    *,
    detail: str = "cute._get_launch_facts()",
) -> LaunchFacts:
    """Validate the compiler's exact dimensions and launch-mode flags.

    Missing facts stay unknown. Upper bounds and user-supplied launch metadata
    are not evidence of an exact launch.
    """

    values = {}
    origins = []
    for field_name in (
        "exact_block_dim",
        "exact_grid_dim",
        "exact_cluster_dim",
        "cooperative_launch",
        "cluster_launch",
    ):
        value = getattr(facts, field_name, None)
        if value is None:
            continue
        if field_name.startswith("exact_"):
            if not isinstance(value, tuple) or len(value) != 3:
                raise ValueError(
                    f"{detail} field {field_name!r} must be a static three-dimensional shape"
                )
            value = normalize_block_dim(value)
            if value is None:
                raise ValueError(
                    f"{detail} field {field_name!r} must contain positive integers"
                )
        elif not isinstance(value, bool):
            raise ValueError(f"{detail} field {field_name!r} must be a bool")
        values[field_name] = value
        origins.append(
            LaunchFactOrigin(
                fact=field_name,
                source="cutlass_provider_api",
                detail=detail,
                verified=True,
            )
        )
    return LaunchFacts(**values, provenance=tuple(origins))


def current_kernel_launch_facts() -> LaunchFacts:
    """Read exact facts, preserving compiler errors for unavailable metadata."""

    runtime = validate_cutlass_runtime()
    return launch_facts_from_cutlass_api(runtime.cute._get_launch_facts())


def current_kernel_block_dim() -> tuple[int, int, int] | None:
    """Return the current kernel's exact block dimensions, if known."""

    return current_kernel_launch_facts().exact_block_dim
