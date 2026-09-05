# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict negative typing fixture for group-first Histogram."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    common_group = coop.this_block()
    common_warp = coop.this_warp()
    common_samples = coop.ThreadData(4, int)
    common_float_samples = coop.ThreadData(4, float)
    common_complex_samples = numba_coop.ThreadData(4, complex)

    class ExternalCounterDtype: ...

    coop.histogram(common_group, 1, bins=8)  # pyright: ignore[reportArgumentType]
    coop.histogram(common_group, common_float_samples, bins=8)  # pyright: ignore[reportArgumentType]
    coop.histogram(common_group, common_complex_samples, bins=8)  # pyright: ignore[reportArgumentType]
    coop.histogram(common_group, common_samples, bins=8, counter_dtype=np.uint8)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.histogram(common_group, common_samples, bins=8, counter_dtype=np.float32)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.histogram(  # pyright: ignore[reportCallIssue]
        common_group,
        common_samples,
        bins=8,
        counter_dtype=ExternalCounterDtype,  # pyright: ignore[reportArgumentType]
    )
    coop.histogram(common_warp, common_samples, bins=8)  # pyright: ignore[reportArgumentType]
    coop.histogram(common_group, common_samples)  # pyright: ignore[reportCallIssue]
    coop.histogram(common_group, common_samples, 8)  # pyright: ignore[reportCallIssue]
    coop.histogram(common_group, common_samples, bins=8, algorithm="tree")  # pyright: ignore[reportArgumentType]
    coop.histogram(common_group, common_samples, bins=8, invented=True)  # pyright: ignore[reportCallIssue]
    coop.histogram(common_group, common_samples, bins="8")  # pyright: ignore[reportArgumentType]
    coop.histogram(common_group, common_samples, bins=8, bins_per_thread="2")  # pyright: ignore[reportArgumentType]

    cutlass_group = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_samples = cutlass_coop.ThreadData(4, int)
    cutlass_float_samples = cutlass_coop.ThreadData(4, float)
    cutlass_complex_samples = cutlass_coop.ThreadData(4, complex)
    cutlass_coop.histogram(cutlass_group, 1.0, bins=8)  # pyright: ignore[reportArgumentType]
    cutlass_coop.histogram(cutlass_group, np.float32(1), bins=8)  # pyright: ignore[reportArgumentType]
    cutlass_coop.histogram(cutlass_group, cutlass_float_samples, bins=8)  # pyright: ignore[reportArgumentType]
    cutlass_coop.histogram(cutlass_group, cutlass_complex_samples, bins=8)  # pyright: ignore[reportArgumentType]
    cutlass_coop.histogram(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_samples,
        bins=8,
        counter_dtype=np.uint8,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.histogram(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_samples,
        bins=8,
        counter_dtype=np.float32,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.histogram(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_samples,
        bins=8,
        counter_dtype=ExternalCounterDtype,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.histogram(cutlass_warp, cutlass_samples, bins=8)  # pyright: ignore[reportArgumentType]
    cutlass_coop.histogram(cutlass_group, cutlass_samples)  # pyright: ignore[reportCallIssue]
    cutlass_coop.histogram(cutlass_group, cutlass_samples, 8)  # pyright: ignore[reportCallIssue]
    cutlass_coop.histogram(cutlass_group, cutlass_samples, bins=8, algorithm="tree")  # pyright: ignore[reportArgumentType]
    cutlass_coop.histogram(cutlass_group, cutlass_samples, bins=8, invented=True)  # pyright: ignore[reportCallIssue]
    cutlass_coop.histogram(cutlass_group, cutlass_samples, bins="8")  # pyright: ignore[reportArgumentType]
    cutlass_coop.histogram(cutlass_group, cutlass_samples, bins=8, bins_per_thread="2")  # pyright: ignore[reportArgumentType]

    numba_group = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_samples = numba_coop.ThreadData(4, int)
    numba_coop.histogram(numba_group, 1, bins=8)  # pyright: ignore[reportArgumentType]
    numba_coop.histogram(numba_warp, numba_samples, bins=8)  # pyright: ignore[reportArgumentType]
    numba_coop.histogram(numba_group, numba_samples)  # pyright: ignore[reportCallIssue]
    numba_coop.histogram(numba_group, numba_samples, 8)  # pyright: ignore[reportCallIssue]
    numba_coop.histogram(numba_group, numba_samples, bins=8, algorithm="tree")  # pyright: ignore[reportArgumentType]
    numba_coop.histogram(numba_group, numba_samples, bins=8, invented=True)  # pyright: ignore[reportCallIssue]
    numba_coop.histogram(numba_group, numba_samples, bins="8")  # pyright: ignore[reportArgumentType]
    numba_coop.histogram(numba_group, numba_samples, bins=8, bins_per_thread="2")  # pyright: ignore[reportArgumentType]
