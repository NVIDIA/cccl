# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict positive typing fixture for group-first Histogram."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import assert_type

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    common_group = coop.this_block()
    common_samples = coop.ThreadData(4, np.uint8)
    assert_type(common_samples, coop.ThreadDataLike[np.uint8])
    assert_type(
        coop.histogram(
            common_group,
            common_samples,
            bins=16,
            algorithm="atomic",
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.histogram(
            common_group,
            common_samples,
            bins=16,
            bins_per_thread=2,
            counter_dtype=np.int32,
            algorithm="sort",
        ),
        coop.ThreadDataLike[np.int32],
    )
    assert_type(
        coop.histogram(
            common_group,
            common_samples,
            bins=16,
            counter_dtype=np.uint32,
        ),
        coop.ThreadDataLike[np.uint32],
    )
    assert_type(
        coop.histogram(
            common_group,
            common_samples,
            bins=16,
            counter_dtype=np.int64,
        ),
        coop.ThreadDataLike[np.int64],
    )
    assert_type(
        coop.histogram(
            common_group,
            common_samples,
            bins=16,
            counter_dtype=np.uint64,
        ),
        coop.ThreadDataLike[np.uint64],
    )
    assert_type(
        coop.histogram(
            common_group,
            common_samples,
            bins=16,
            counter_dtype=int,
        ),
        coop.ThreadDataLike[int],
    )

    class ExternalCounterDtype: ...

    external_counter_dtype: Any = ExternalCounterDtype

    assert_type(
        coop.histogram(
            common_group,
            common_samples,
            bins=16,
            counter_dtype=external_counter_dtype,
        ),
        coop.ThreadDataLike[Any],
    )

    cutlass_group = cutlass_coop.this_block()
    cutlass_samples = cutlass_coop.ThreadData(4, np.uint8)
    assert_type(cutlass_samples, cutlass_coop.ThreadData[np.uint8])
    assert_type(
        cutlass_coop.histogram(
            cutlass_group,
            cutlass_samples,
            bins=16,
            algorithm="atomic",
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.histogram(
            cutlass_group,
            cutlass_samples,
            bins=16,
            bins_per_thread=2,
            counter_dtype=np.int32,
            algorithm="sort",
        ),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(
        cutlass_coop.histogram(
            cutlass_group,
            cutlass_samples,
            bins=16,
            counter_dtype=np.uint32,
        ),
        cutlass_coop.ThreadData[np.uint32],
    )
    assert_type(
        cutlass_coop.histogram(
            cutlass_group,
            cutlass_samples,
            bins=16,
            counter_dtype=np.int64,
        ),
        cutlass_coop.ThreadData[np.int64],
    )
    assert_type(
        cutlass_coop.histogram(
            cutlass_group,
            cutlass_samples,
            bins=16,
            counter_dtype=np.uint64,
        ),
        cutlass_coop.ThreadData[np.uint64],
    )
    assert_type(
        cutlass_coop.histogram(cutlass_group, 1, bins=16),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.histogram(
            cutlass_group,
            cutlass_samples,
            bins=16,
            counter_dtype=int,
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.histogram(
            cutlass_group,
            cutlass_samples,
            bins=16,
            counter_dtype=external_counter_dtype,
        ),
        cutlass_coop.ThreadData[Any],
    )

    numba_group = numba_coop.this_block()
    numba_samples = numba_coop.ThreadData(4, np.uint8)
    assert_type(numba_samples, coop.ThreadDataLike[np.uint8])
    assert_type(
        numba_coop.histogram(
            numba_group,
            numba_samples,
            bins=16,
            algorithm="atomic",
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.histogram(
            numba_group,
            numba_samples,
            bins=16,
            bins_per_thread=2,
            counter_dtype=np.int32,
            algorithm="sort",
        ),
        coop.ThreadDataLike[np.int32],
    )
    assert_type(
        numba_coop.histogram(
            numba_group,
            numba_samples,
            bins=16,
            counter_dtype=np.uint32,
        ),
        coop.ThreadDataLike[np.uint32],
    )
    assert_type(
        numba_coop.histogram(
            numba_group,
            numba_samples,
            bins=16,
            counter_dtype=np.int64,
        ),
        coop.ThreadDataLike[np.int64],
    )
    assert_type(
        numba_coop.histogram(
            numba_group,
            numba_samples,
            bins=16,
            counter_dtype=np.uint64,
        ),
        coop.ThreadDataLike[np.uint64],
    )
    assert_type(
        numba_coop.histogram(
            numba_group,
            numba_samples,
            bins=16,
            counter_dtype=int,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.histogram(
            numba_group,
            numba_samples,
            bins=16,
            algorithm=numba_coop.BlockHistogramAlgorithm.ATOMIC,
        ),
        coop.ThreadDataLike[int],
    )
