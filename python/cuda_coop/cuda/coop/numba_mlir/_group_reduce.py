# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Reduce and Sum markers for Numba-CUDA-MLIR."""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation(
    "reduce",
    family_module="cuda.coop.numba_mlir._compiler._group_reduce",
)
def reduce(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    binary_op: Any = None,
    broadcast: Any = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Reduce values across a group."""

    return group_primitive_marker(
        "reduce",
        group,
        value,
        binary_op=binary_op,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


@group_operation(
    "sum",
    family_module="cuda.coop.numba_mlir._compiler._group_reduce",
)
def sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    broadcast: Any = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Sum values across a group."""

    return group_primitive_marker(
        "sum",
        group,
        value,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


__all__ = ["reduce", "sum"]
