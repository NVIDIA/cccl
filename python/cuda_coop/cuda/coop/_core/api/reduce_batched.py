# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declare batched reductions for the common kernel API.

The registration lets compiler backends recognize this function by identity.
The Python body rejects host calls; the backend supplies the device operation.
"""

from __future__ import annotations

from typing import Literal, TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ReduceOperator,
)

from ..thread_group import CoopCompilerContextRequiredError
from ._dispatch import (
    _common_group_operation,
)
from ._payload import (
    ThreadDataLike,
)
from .thread_group import WarpGroup

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)


@_common_group_operation(
    "reduce_batched", group_kinds=("warp", "threads_within_warp")
)
def reduce_batched(
    group: WarpGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    binary_op: ReduceOperator | None = None,
    output_layout: Literal["striped", "blocked"] = "striped",
) -> ThreadDataLike[_ItemT]:
    """Reduce each payload slot independently across the selected warp.

    Parameters
    ----------
    group : ThreadGroup
        A physical warp or a logical warp from ``threads_within_warp``.
        Every member of the selected warp participates in the same call.
    value : ThreadDataLike
        Readable payload with a positive compile-time extent ``B``. Local
        slot ``b`` contributes to batch ``b`` across all lanes. Dtypes are
        signed and unsigned 8-, 16-, 32-, and 64-bit integers, float32, and
        float64.
    binary_op : str, optional
        Built-in reduction operator, default addition. Operator strings
        follow :func:`cuda.coop.reduce`, including sum, multiplication,
        minimum, maximum, and integer bitwise operations. The operator must
        be associative and commutative.
    output_layout : {"striped", "blocked"}, optional
        Compile-time ownership layout, default ``"striped"``. For ``W``
        lanes, local result slot ``i`` in lane ``r`` holds batch
        ``r + i * W`` in striped layout, or
        ``r * ceil(B / W) + i`` in blocked layout.

    Returns
    -------
    ThreadDataLike
        A fresh payload of ``ceil(B / W)`` aggregates per lane, using the
        input dtype without accumulator promotion. Slots without a batch
        are unspecified. Guard reads and stores using the batch index.
        The input is preserved.

    Notes
    -----
    Each batch reduces independently; input slots are not combined with
    one another. The Numba-CUDA-MLIR backend supports complete physical
    warps and logical warps of 1, 2, 4, 8, or 16 threads. The compiler manages
    scratch per warp; this operation has no ``temp_storage`` argument.

    Use :func:`cuda.coop.numba_mlir.reduce_batched` for a custom stateless
    device operator. The CUB counterpart is ``cub::WarpReduceBatched``.

    Examples
    --------
    Sum each feature independently across a warp with Numba-CUDA-MLIR. Each
    warp reads its own 32 rows, and the first lanes store the feature sums.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_reduce_batched.py
        :language: python
        :start-after: # example-begin reduce-batched-features
        :end-before: # example-end reduce-batched-features
        :dedent: 4
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.reduce_batched must be called from a supported GPU kernel."
    )


__all__ = ["reduce_batched"]
