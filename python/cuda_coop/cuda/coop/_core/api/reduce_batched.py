# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common independent-batch reduction entry point."""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _common_group_operation,
    _common_selector,
    _group_primitive_marker,
)
from ._payload import (
    ThreadDataLike,
    _ReadableThreadDataLike,
)
from .reduce import _common_reduce_operator, _validate_common_reduce_value


@_common_group_operation("reduce_batched", group_kinds=("warp", "threads_within_warp"))
def reduce_batched(
    group: ThreadGroup,
    value: _ReadableThreadDataLike[Any],
    /,
    *,
    binary_op: Any = None,
    output_layout: str = "striped",
) -> ThreadDataLike[Any]:
    """Reduce each payload slot independently across the selected warp.

    Implemented by Numba-CUDA-MLIR. The CUTLASS backend does not currently
    support this operation.

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
    """

    output_layout = _common_selector(
        "reduce_batched", "output_layout", output_layout, {"striped", "blocked"}
    )
    binary_op = _common_reduce_operator(binary_op)
    if _backend_module_name() is not None:
        if not isinstance(value, _ReadableThreadDataLike):
            raise TypeError("cuda.coop.reduce_batched requires a ThreadData payload")
        _validate_common_reduce_value("reduce_batched", value, binary_op)
    return _group_primitive_marker(
        "reduce_batched",
        group,
        value,
        binary_op=binary_op,
        output_layout=output_layout,
    )


__all__ = ["reduce_batched"]
