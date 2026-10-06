# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose the batched-reduction marker and qualified operator extension.

The compiler loads this operation's planning family on demand. Each input
slot reduces across the selected warp, and layout distributes those batch
results among lanes. The marker body does not perform Python reductions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeVar

import numpy

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ReduceOperator,
    ThreadDataLike,
)

from .._compiler._operations import group_operation
from .._thread_group import WarpGroup
from ._marker import group_primitive_marker

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)


@group_operation(
    "reduce_batched",
    family_module="cuda.coop.numba_mlir._compiler._group_reduce_batched",
)
def reduce_batched(
    group: WarpGroup,
    value: CommonThreadDataLike[_ItemT] | numpy.ndarray,
    /,
    *,
    binary_op: ReduceOperator
    | Callable[[_ItemT, _ItemT], _ItemT]
    | None = None,
    output_layout: Literal["striped", "blocked"] = "striped",
) -> ThreadDataLike[_ItemT]:
    """Reduce independent payload slots across a physical or logical warp.

    Extends :func:`cuda.coop.reduce_batched` with local-array inputs and a
    stateless device ``binary_op`` callback. The operator must be associative
    and commutative. Inputs are preserved; output slots without a batch are
    unspecified. The default operator is addition.

    Examples
    --------
    Reduce each feature independently across a warp using a device maximum
    operator.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_reduce_batched_examples.py
        :language: python
        :start-after: # qualified-batched-reduce-example-begin
        :end-before: # qualified-batched-reduce-example-end
        :dedent: 4
    """

    return group_primitive_marker(
        "reduce_batched",
        group,
        value,
        binary_op=binary_op,
        output_layout=output_layout,
    )


__all__ = ["reduce_batched"]
