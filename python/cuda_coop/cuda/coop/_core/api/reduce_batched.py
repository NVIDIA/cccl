# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable independent-batch reduction entry point."""

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup
from ._dispatch import (
    _portable_group_operation,
)
from ._payload import (
    ThreadDataLike,
    _ReadableThreadDataLike,
)


@_portable_group_operation(
    "reduce_batched", group_kinds=("warp", "threads_within_warp")
)
def reduce_batched(
    group: ThreadGroup,
    value: _ReadableThreadDataLike[Any],
    /,
    *,
    binary_op: Any = None,
    output_layout: str = "striped",
) -> ThreadDataLike[Any]:
    """Reduce each payload slot independently across the selected warp.

    If each lane supplies ``B`` items and the warp contains ``W`` lanes, the
    fresh result contains ``ceil(B / W)`` items per lane. Striped output slot
    ``i`` in lane ``r`` holds batch ``r + i * W``; blocked output holds batch
    ``r * ceil(B / W) + i``. Slots beyond ``B`` are unspecified. Inputs are
    preserved. Every member of the selected warp must participate.

    ``B`` must be a positive compile-time extent. ``binary_op`` defaults to
    addition and accepts the same built-in string operators as :func:`reduce`.
    The operation must be associative and commutative. Input and output share
    the same numeric dtype; there is no accumulator promotion. Supported dtypes
    are signed and unsigned 8-, 16-, 32-, and 64-bit integers, ``float32``, and
    ``float64``.

    The Numba-CUDA-MLIR backend supports complete physical warps and logical
    warps of 1, 2, 4, 8, or 16 threads. Temporary storage is managed per warp
    by the compiler. Guard stores using the batch index, since the allocated
    result extent may include slots without a corresponding batch.
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.reduce_batched must be called from a supported GPU kernel."
    )


__all__ = ["reduce_batched"]
