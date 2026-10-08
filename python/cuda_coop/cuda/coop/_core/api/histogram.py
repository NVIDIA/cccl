# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose fresh block histograms through the common compiled-kernel API.

Samples contain bin indices. The result is a separate counter payload whose
dtype and per-thread extent can differ from the input. The decorator registers
this function so that compilers can recognize its calls. An ordinary Python
call raises a compiler-context error. Each call zeroes its shared counters
before counting, so reused scratch never carries counts from an earlier call.
To accumulate tiles, add the returned counters yourself.
"""

from __future__ import annotations

from cuda.coop._typing import CompilerIntegerLike

try:
    import numpy
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise


from typing import Literal, TypeVar

from cuda.coop._typing import (
    CommonThreadDataLike,
    ThreadDataLike,
)

from ..thread_group import CoopCompilerContextRequiredError
from ._dispatch import (
    _common_group_operation,
)
from ._payload import (
    TempStorageLike,
)
from .thread_group import BlockGroup

_Counter = TypeVar(
    "_Counter", "numpy.int32", "numpy.uint32", "numpy.int64", "numpy.uint64"
)


@_common_group_operation("histogram", group_kinds=("block",))
def histogram(
    group: BlockGroup,
    samples: CommonThreadDataLike[
        int
        | numpy.uint8
        | numpy.int32
        | numpy.uint32
        | numpy.int64
        | numpy.uint64
        | CompilerIntegerLike
    ],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[int | _Counter] | numpy.dtype | None = None,
    algorithm: Literal["atomic", "sort"] = "atomic",
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[numpy.int32] | ThreadDataLike[_Counter]:
    """Return fresh striped bin counts, preserving the input samples.

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
        the block before reusing the descriptor in another collective.

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

    With ``algorithm="atomic"``, CUB needs no algorithm scratch. This
    operation still uses shared memory for intermediate bin counters before
    returning the per-thread counts. ``bins`` specifies the number of
    counters; it does not supply their storage. Omit ``temp_storage`` to
    allocate that storage automatically.

    There is no ``valid_items`` control. Zero-padding an incomplete input
    tile adds counts to bin zero. The input tile size and projected output
    size must fit signed 32-bit integers. The CUB counterpart is
    ``cub::BlockHistogram``.

    Examples
    --------
    Accumulate three complete input tiles with Numba-CUDA-MLIR. Each call
    returns fresh counts, which the kernel adds to int64 totals. The striped
    Store writes bins in order and omits the extra output slots.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_histogram_examples.py
        :language: python
        :start-after: # histogram-accumulation-example-begin
        :end-before: # histogram-accumulation-example-end
        :dedent: 4
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.histogram must be called from a supported GPU kernel."
    )


__all__ = ["histogram"]
