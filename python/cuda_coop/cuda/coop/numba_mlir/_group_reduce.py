# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first scalar reduction markers for Numba-CUDA-MLIR."""

from __future__ import annotations

from typing import Any, TypeVar

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup

_ScalarT = TypeVar("_ScalarT")


@group_operation("reduce")
def reduce(
    group: ThreadGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: Any = None,
    valid_items: Any = None,
    algorithm: Any = None,
) -> _ScalarT:
    """Reduce one scalar per group member and return the root result.

    Every group member must participate in converged control flow, and only
    group rank zero may consume the result. A runtime ``valid_items`` must be
    uniform, positive, and no larger than the group size; it is converted to
    CUB's ``int`` parameter by the direct provider. ``algorithm`` applies only
    to block groups.
    """

    return group_primitive_marker(
        "reduce",
        group,
        value,
        binary_op=binary_op,
        valid_items=valid_items,
        algorithm=algorithm,
    )


@group_operation("sum")
def sum(
    group: ThreadGroup,
    value: _ScalarT,
    /,
    *,
    valid_items: Any = None,
    algorithm: Any = None,
) -> _ScalarT:
    """Sum one scalar per group member and return the root result.

    Every group member must participate in converged control flow, and only
    group rank zero may consume the result. A runtime ``valid_items`` must be
    uniform, positive, and no larger than the group size; it is converted to
    CUB's ``int`` parameter by the direct provider. ``algorithm`` applies only
    to block groups.
    """

    return group_primitive_marker(
        "sum",
        group,
        value,
        valid_items=valid_items,
        algorithm=algorithm,
    )


__all__ = ["reduce", "sum"]
