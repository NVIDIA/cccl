# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup
from ._dispatch import (
    _common_group_operation,
)
from ._payload import (
    TempStorageLike,
)


@_common_group_operation("histogram", group_kinds=("block",))
def histogram(
    group: ThreadGroup,
    samples: Any,
    /,
    *,
    bins: Any,
    bins_per_thread: Any = 1,
    counter_dtype: object | None = None,
    algorithm: str = "atomic",
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Return fresh striped bin counts, preserving the input samples.

    Implemented by both Numba-CUDA-MLIR and CUTLASS.

    Parameters
    ----------
    group : ThreadGroup
        A complete one-dimensional ``this_block()`` group. Every member
        participates in the same call.
    samples : ThreadDataLike
        Readable fixed-size payload of bin indices in ``[0, bins)``. Dtypes
        uint8, int32, uint32, int64, and uint64 are supported. Every input
        slot contributes one sample.
    bins : int
        Positive compile-time number of bins.
    bins_per_thread : int, optional
        Positive compile-time result extent, default one. The product
        ``block_size * bins_per_thread`` must cover every bin.
    counter_dtype : object, optional
        int32 by default, or uint32, int64, or uint64, independently of the
        sample dtype. The Python ``int`` spelling means int32.
    algorithm : {"atomic", "sort"}, optional
        Compile-time counting algorithm, default ``"atomic"``. Both preserve
        the input samples; the sort path operates on a private copy.
    temp_storage : TempStorageLike, optional
        Explicit shared scratch for CUB storage and intermediate counters.
        Omit it for automatic storage. With ``auto_sync=False``, synchronize
        the block before reusing the descriptor in another primitive.

    Returns
    -------
    ThreadDataLike
        A fresh payload with ``bins_per_thread`` counters per member. Member
        ``t`` receives bin ``t + i * block_size`` in local slot ``i``. Slots
        beyond ``bins`` contain zero. Use striped Store to write bin order.

    Notes
    -----
    Each call initializes its counters to zero, including when scratch is
    reused. Accumulate tiles by adding corresponding returned counters and
    choose a dtype wide enough for the accumulated total. No running
    histogram is retained in ``TempStorage``.

    There is no ``valid_items`` control. Zero-padding an incomplete input
    tile adds counts to bin zero. The input tile size and projected output
    size must fit signed 32-bit integers. The CUB counterpart is
    ``cub::BlockHistogram``. Use
    :func:`cuda.coop.numba_mlir.histogram` for scalar or local-array samples.
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.histogram must be called from a supported GPU kernel."
    )


__all__ = ["histogram"]
