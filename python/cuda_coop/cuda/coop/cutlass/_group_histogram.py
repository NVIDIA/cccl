# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fresh block histogram counters for CuTe kernels."""

from cuda.coop._core.block._common import normalize_positive_int
from cuda.coop._core.block.histogram import normalize_histogram_algorithm
from cuda.coop._core.thread_group import ThreadGroup

from ._temp_storage import TempStorage
from ._thread_data import _snapshot_readable_payload


def histogram(
    group,
    samples,
    /,
    *,
    bins,
    bins_per_thread=1,
    counter_dtype=None,
    algorithm="atomic",
    temp_storage=None,
):
    """Count integer samples into fresh striped bin counters.

    Parameters
    ----------
    group : ThreadGroup
        A complete one-dimensional block. Every thread participates.
    samples : ThreadData or CuTe register payload
        Readable fixed-size bin indices in ``[0, bins)``. Supports uint8,
        int32, uint32, int64, and uint64. CuTe register tensors and immutable
        vectors are copied through
        :meth:`ThreadData.from_payload <cuda.coop.cutlass.ThreadData.from_payload>`.
    bins : int
        Positive compile-time number of bins.
    bins_per_thread : int, optional
        Positive compile-time result extent. ``block_size * bins_per_thread``
        must cover every bin. Defaults to one.
    counter_dtype : object, optional
        Independent counter dtype: int32 (default), uint32, int64, or uint64.
        Accepts NumPy and CuTe selectors. Python ``int`` means int32.
    algorithm : {"atomic", "sort"}, optional
        Compile-time algorithm. Both preserve the input samples.
    temp_storage : TempStorage, optional
        Explicit scratch for CUB storage and intermediate counters. Requested
        alignment is a minimum. With ``auto_sync=False``, synchronize the
        block before reusing the descriptor.

    Returns
    -------
    ThreadData
        Fresh counters in striped order: member ``t`` receives bin
        ``t + i * block_size`` in slot ``i``. Slots beyond ``bins`` are zero.
        Use striped Store to write contiguous bin order.

    Notes
    -----
    Each call starts from zero, even when scratch is reused. Accumulate
    returned counters explicitly to count several tiles. There is no
    ``valid_items`` control; padding contributes samples. The complete
    contract is shared with :func:`cuda.coop.histogram`. See the
    :doc:`Histogram visualization <coop/visualizations/histogram>` for sample
    and counter layouts.
    """
    if not isinstance(group, ThreadGroup):
        raise TypeError("cuda.coop.cutlass.histogram group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError("cuda.coop.cutlass.histogram requires a block group")
    bins = normalize_positive_int("bins", bins)
    bins_per_thread = normalize_positive_int("bins_per_thread", bins_per_thread)
    algorithm = normalize_histogram_algorithm(algorithm)
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError("histogram temp_storage must be CUTLASS TempStorage")
    samples = _snapshot_readable_payload(samples, name="samples", primitive="histogram")
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
