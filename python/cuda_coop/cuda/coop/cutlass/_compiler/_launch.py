# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Read and validate CuTe kernel dimensions and launch flags."""

from __future__ import annotations

import operator
from typing import Any

from ..._core.launch import LaunchFactOrigin, LaunchFacts
from ._runtime import validate_cutlass_runtime


def normalize_block_dim(value: Any) -> tuple[int, int, int] | None:
    """Normalize positive static dimensions and pad to three with ones.

    Return None for unsupported shapes, floats, or Boolean dimensions. Dynamic
    DSL values cannot establish an exact launch dimension.
    """

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


def launch_facts_from_cutlass_api(
    facts: Any,
    *,
    detail: str = "cute._get_launch_facts()",
) -> LaunchFacts:
    """Validate the compiler's exact dimensions and launch-mode flags.

    ``current_kernel_launch_facts`` calls this during primitive lowering.
    The shared planner needs actual launch dimensions to instantiate CUB
    types and check that every required thread belongs to the group.

    Missing facts stay unknown. Upper bounds and user-supplied launch metadata
    are not evidence of an exact launch.

    Parameters
    ----------
    facts : object
        Result of CuTe's launch-facts query. Absent fields remain unknown.
    detail : str
        Query name recorded in the origin of each fact and in diagnostics.

    Returns
    -------
    LaunchFacts
        Validated dimensions and flags with compiler-source annotations.
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
                    f"{detail} field {field_name!r} must be a static "
                    "three-dimensional shape"
                )
            value = normalize_block_dim(value)
            if value is None:
                raise ValueError(
                    f"{detail} field {field_name!r} must contain "
                    "positive integers"
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
