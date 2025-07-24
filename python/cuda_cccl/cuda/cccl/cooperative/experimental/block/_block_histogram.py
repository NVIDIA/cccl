# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
cuda.cccl.cooperative.block.histogram
=====================================

This module provides a set of :ref:`collective <collective-primitives>`
methods for constructing block-wide histograms from data samples partitioned
across CUDA thread blocks.

It contains an implementation of the "one-shot" `cub::BlockHistogram`
primitive.
"""

from enum import IntEnum, auto
from typing import Callable

import numba

from .._common import (
    make_binary_tempfile,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import (
    Algorithm,
    Dependency,
    DependentArray,
    Invocable,
    Pointer,
    TemplateParameter,
)
from .._typing import (
    DimType,
    DtypeType,
)


class BlockHistogramAlgorithm(IntEnum):
    """
    Enum representing the different algorithms available for block histogram.
    """

    SORT = auto()
    ATOMIC = auto()

    def __str__(self):
        return f"::cub::BLOCK_HISTO_{self.name.upper()}"


def histogram(
    item_dtype: DtypeType,
    counter_dtype: DtypeType,
    dim: DimType,
    items_per_thread: int,
    bins: int,
    algorithm: BlockHistogramAlgorithm = None,
) -> Callable:
    """
    Creates a block-wide histogram primitive based on the CUB library's
    BlockHistogram functionality.

    :param item_dtype: Supplies the data type of the items being sampled.
        Note that this is almost always an 8-bit integer type, e.g.
        ``np.uint8``.

    :param counter_dtype: Supplies the data type of the histogram counters.

    :param dim: Supplies the dimensions of the CUDA thread block.

    :param items_per_thread: Supplies the number of items each thread will
        sample.  A good default to start with is usually 4.

    :param bins: Supplies the number of bins in the histogram.  This should
        represent the number of unique values that can be sampled by the
        ``item_dtype``, i.e. if ``item_dtype`` is ``np.uint8``, then ``bins``
        should be 256.

    :param algorithm: Supplies the algorithm to use for the histogram.
        Defaults to ``BlockHistogramAlgorithm.ATOMIC``, which uses atomic
        operations to build the histogram.

    :raises ValueError: If ``items_per_thread`` is less than 1.

    :raises ValueError: If ``algorithm`` is not one of the
        :py:class:`BlockHistogramAlgorithm` enum values.

    :raises ValueError: If ``bins`` is not a positive integer.

    :returns A callable that takes two array arguments: the first is the array
        of items to be sampled, and the second is the array of histogram bins.

    """

    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")

    if algorithm is None:
        algorithm = BlockHistogramAlgorithm.ATOMIC

    if algorithm not in BlockHistogramAlgorithm:
        raise ValueError(
            f"Invalid algorithm: {algorithm!r}. "
            f"Must be one of {list(BlockHistogramAlgorithm)}."
        )

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(f"Invalid bins: {bins!r}. Must be a positive integer.")

    dim = normalize_dim_param(dim)
    item_dtype = normalize_dtype_param(item_dtype)
    counter_dtype = normalize_dtype_param(counter_dtype)

    template_parameters = [
        TemplateParameter("T"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("ITEMS_PER_THREAD"),
        TemplateParameter("BINS"),
        TemplateParameter("ALGORITHM"),
        TemplateParameter("BLOCK_DIM_Y"),
        TemplateParameter("BLOCK_DIM_Z"),
    ]

    parameters = [
        [
            Pointer(numba.uint8),
            DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
            DependentArray(Dependency("CounterT"), Dependency("BINS")),
        ]
    ]

    specialization_kwds = {
        "T": item_dtype,
        "BLOCK_DIM_X": dim[0],
        "ITEMS_PER_THREAD": items_per_thread,
        "BINS": bins,
        "ALGORITHM": str(algorithm),
        "BLOCK_DIM_Y": dim[1],
        "BLOCK_DIM_Z": dim[2],
        "CounterT": counter_dtype,
    }

    struct_name = "BlockHistogram"
    method_name = "Histogram"
    c_name = "block_histogram"
    includes = ["cub/block/block_histogram.cuh"]

    template = Algorithm(
        struct_name,
        method_name,
        c_name,
        includes,
        template_parameters,
        parameters,
    )

    specialization = template.specialize(specialization_kwds)

    return Invocable(
        temp_files=[
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in specialization.get_lto_ir()
        ],
        temp_storage_bytes=specialization.temp_storage_bytes,
        temp_storage_alignment=specialization.temp_storage_alignment,
        algorithm=specialization,
    )
