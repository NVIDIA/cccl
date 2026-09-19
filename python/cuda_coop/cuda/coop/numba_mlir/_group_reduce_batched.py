# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Batched warp reduction marker for Numba-CUDA-MLIR."""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation(
    "reduce_batched",
    family_module="cuda.coop.numba_mlir._compiler._group_reduce_batched",
)
def reduce_batched(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    binary_op: Any = None,
    output_layout: str = "striped",
) -> Any:
    """Reduce independent payload slots across a physical or logical warp.

    Extends :func:`cuda.coop.reduce_batched` with local-array inputs and a
    stateless device ``binary_op`` callback. The operator must be associative
    and commutative. Inputs are preserved; output slots without a batch are
    unspecified. The default operator is addition.
    """

    return group_primitive_marker(
        "reduce_batched",
        group,
        value,
        binary_op=binary_op,
        output_layout=output_layout,
    )


__all__ = ["reduce_batched"]
