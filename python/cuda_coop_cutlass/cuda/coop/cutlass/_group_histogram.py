# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A histogram counts how many of a block's samples fall in each bin.

Each sample is itself a bin index: a sample with value ``b`` adds one to bin
``b``. To count values of another kind, such as floats in a range, compute
each value's bin index first.

.. figure:: visualizations/histogram.svg
   :alt: Eight threads hold two samples each, sixteen bin indices from 0 to
         12. A row of thirteen bins shows the counts 2, 1, 1, 2, 1, 1, 2,
         0, 1, 2, 1, 1, 1. Below it each thread holds two counters. Thread
         0 holds the counts of bins 0 and 8, and thread 4 those of bins 4
         and 12. Threads 5, 6 and 7 each hold the count of one bin and a
         zero marked as padding.
   :width: 100%

   ``histogram`` with ``bins=13`` and ``bins_per_thread=2`` on a block of
   eight threads with two samples each. Thread 0 receives the counts of
   bins 0 and 8, which are 2 and 1. The second counters of threads 5, 6 and
   7 stand for bins 13, 14 and 15, which are past the last bin, and hold
   zero.

Each call counts from zero, even when it shares a
:ref:`TempStorage <coop-temp-storage>` with earlier calls. To count several
:term:`tiles <tile>`, add each call's counters to a payload of your own.
Item ``i`` of a thread's counters stands for the same bin in every call, so
each thread adds item by item, as in the second example of
:func:`~cuda.coop.cutlass.histogram`.

``histogram`` counts every item of every thread and takes no
``valid_items``, so zeros that fill a partial tile add to bin 0. To count a
partial tile, fill it with the index of an extra bin, as in
:ref:`this FAQ <coop-faq-histogram-padding>`.
"""

# Implementation notes.
#
# Count block-wide integer samples into bins for CuTe kernels.
#
# Check the group, bin count, output extent, algorithm, and scratch here.
# These controls are fixed at compile time. Copy samples into a new
# ThreadData so read-only payloads and CuTe register tensors or TensorSSA
# values are accepted. The lowering provider then resolves the dtypes and
# emits the call.

from __future__ import annotations

from typing import TypeVar

from cuda.coop.cutlass._planning.block._common import normalize_positive_int
from cuda.coop.cutlass._planning.block.histogram import (
    normalize_histogram_algorithm,
)
from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import CompilerIntegerLike

from ._temp_storage import TempStorage
from ._thread_data import _snapshot_readable_payload

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
from typing import Literal

from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
)
from ._typing import BlockGroup, ReadableThreadDataLike

_Counter = TypeVar(
    "_Counter",
    "np.int32",
    "np.uint32",
    "np.int64",
    "np.uint64",
    "CompilerIntegerLike",
)


def histogram(
    group: BlockGroup,
    samples: ReadableThreadDataLike[
        int
        | np.uint8
        | np.int32
        | np.uint32
        | np.int64
        | np.uint64
        | CompilerIntegerLike
    ]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[int | _Counter] | np.dtype | None = None,
    algorithm: Literal["atomic", "sort"] = "atomic",
    temp_storage: TempStorage | None = None,
) -> ThreadData:
    """Count how many of a block's samples equal each bin index.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, on a one-dimensional block of any size. Every
        thread of the block makes the call, on every trip of a loop that
        contains it. Any other group is rejected when the kernel is
        compiled, and so is a block with two or three dimensions.
    samples : ThreadData, CuTe register tensor or TensorSSA
        The bin indices, one in each item of a payload. The element type is
        ``Uint8``, ``Int32``, ``Uint32``, ``Int64`` or ``Uint64``. Another
        type is rejected when the kernel is compiled, and
        :meth:`~cuda.coop.cutlass.ThreadData.from_fn` converts a payload to
        one of the five. For one sample per thread, pass a payload of one
        item. A register tensor or ``TensorSSA`` is read as a payload.

        Each sample must be at least 0 and less than ``bins``. Nothing
        checks this when the kernel is compiled or while it runs. A sample
        outside the range is still used as an index. With ``"atomic"`` it
        adds one to memory outside the counters. With ``"sort"`` it can
        change the counts of other bins or overwrite other memory.
    bins : int
        The number of bins, a positive Python integer.
    bins_per_thread : int, optional
        The number of counters each thread receives, a positive Python
        integer. The default is 1. The block size times ``bins_per_thread``
        must be at least ``bins``, so the default needs a block of at least
        ``bins`` threads. A smaller product is rejected when the kernel is
        compiled. ``bins_per_thread`` can differ from the number of samples
        per thread.
    counter_dtype : type, optional
        The element type of the counters: ``Int32``, ``Uint32``, ``Int64``
        or ``Uint64``, given as a CUTLASS type, the matching NumPy type or
        a NumPy ``dtype``. The default is ``Int32``, and Python ``int``
        means ``Int32`` too. Another type is rejected when the kernel is
        compiled. The choice is independent of the element type of
        ``samples``. The counts of one call fit all four types. When the
        kernel adds the counts of several calls, choose a type that holds
        the totals.
    algorithm : {"atomic", "sort"}, optional
        A Python string. ``"atomic"``, the default, adds one to a bin's
        counter for each sample with an atomic addition. ``"sort"``
        radix-sorts a copy of the samples and counts each run of equal
        values. Both give the same counts. With ``"sort"``, the block size
        times the item count of ``samples`` is at most 65,535. A larger tile
        is rejected when the kernel is compiled. CUB's documentation says
        that ``"sort"`` has consistent throughput for any distribution of
        the samples, and that ``"atomic"`` can slow down when many samples
        fall in the same bin. CUB's own default is ``"sort"``.
    temp_storage : TempStorage, optional
        Both algorithms keep one counter per bin in shared memory. The
        ``"sort"`` algorithm also uses scratch to radix-sort the samples,
        then reuses that space to find the start and end of each bin's run.
        Leave this out and the compiler provides the scratch and a barrier
        after the call. Pass a :ref:`TempStorage <coop-temp-storage>` to
        share the bytes with other calls. A barrier then follows the call
        only if the ``TempStorage`` was created with ``auto_sync=True``.

    Returns
    -------
    cuda.coop.cutlass.ThreadData
        This thread's counters, a new payload of ``bins_per_thread`` items
        in the element type that ``counter_dtype`` selects. They are in
        :term:`striped` order: on a block of ``G`` threads, item ``i`` of
        the thread with rank ``t`` holds the count of bin ``t + i * G``.
        Every item is defined, and an item whose bin index is ``bins`` or
        more holds zero. ``samples`` keeps its values. When ``samples`` is a
        ``ThreadData``, the result has its alignment.

        :func:`~cuda.coop.cutlass.store` with ``algorithm="striped"`` writes
        the counters to memory in bin order. It writes ``bins_per_thread``
        elements for each thread of the block, the zeros included. Pass it
        ``valid_items=bins`` to write the counts alone.

    Notes
    -----
    The call runs CUB's ``Histogram`` method. CUB's ``Composite`` method,
    which adds to counters the caller keeps, has no counterpart here. The
    second example below does its job.

    See Also
    --------
    :cpp:struct:`cub::BlockHistogram`
        The CUB class behind the call.

    Examples
    --------
    Count one tile into 70 bins, once with each algorithm.
    ``atomic_counts`` and ``sort_counts`` hold the same counts. The block
    has 64 threads and each thread receives two counters, so each ``store``
    writes 128 elements: the 70 counts in bin order, then 58 zeros.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_histogram_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_histogram_documentation.py
        :language: python
        :start-after: # cutlass-histogram-example-begin
        :end-before: # cutlass-histogram-example-end
        :dedent: 4

    Count ``tiles`` tiles of 256 samples each and add up their counts.
    ``totals`` starts at zero, and each thread adds the counters of every
    call to it, item by item.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_histogram_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_histogram_documentation.py
        :language: python
        :start-after: # cutlass-histogram-tiles-example-begin
        :end-before: # cutlass-histogram-tiles-example-end
        :dedent: 4
    """
    if not isinstance(group, ThreadGroup):
        raise TypeError(
            "cuda.coop.cutlass.histogram group must be a ThreadGroup"
        )
    if group.kind != "block":
        raise NotImplementedError(
            "cuda.coop.cutlass.histogram requires a block group"
        )
    bins = normalize_positive_int("bins", bins)
    bins_per_thread = normalize_positive_int("bins_per_thread", bins_per_thread)
    algorithm = normalize_histogram_algorithm(algorithm)
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError("histogram temp_storage must be CUTLASS TempStorage")
    samples = _snapshot_readable_payload(
        samples, name="samples", primitive="histogram"
    )
    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._histogram import provider_histogram

    return provider_histogram(
        group=group,
        launch=current_kernel_launch_facts(),
        samples=samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=counter_dtype,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


__all__ = ["histogram"]
