# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict negative typing fixture for group-first Radix Sort."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    block = coop.this_block()
    warp = coop.this_warp()
    keys = coop.ThreadData(2, int)
    float_keys = coop.ThreadData(2, float)
    narrow_keys = numba_coop.ThreadData(2, np.int16)
    values = coop.ThreadData(2, np.float32)
    complex_values = numba_coop.ThreadData(2, complex)
    opaque: object = object()

    # The common profile is block-only and accepts only portable integral
    # ThreadData payloads, not qualified scalar or register forms.
    coop.radix_sort_keys(block, 1)  # pyright: ignore[reportArgumentType]
    coop.radix_sort_keys(block, np.int32(1))  # pyright: ignore[reportArgumentType]
    coop.radix_sort_keys(block, float_keys)  # pyright: ignore[reportArgumentType]
    coop.radix_sort_keys(block, narrow_keys)  # pyright: ignore[reportArgumentType]
    coop.radix_sort_keys(warp, keys)  # pyright: ignore[reportArgumentType]

    # Bit positions are Python, NumPy, or compiler integers; order is a bool.
    coop.radix_sort_keys(block, keys, begin_bit=1.5)  # pyright: ignore[reportArgumentType]
    coop.radix_sort_keys(block, keys, begin_bit=opaque)  # pyright: ignore[reportArgumentType]
    coop.radix_sort_keys(block, keys, end_bit=np.float32(16))  # pyright: ignore[reportArgumentType]
    coop.radix_sort_keys(block, keys, end_bit=opaque)  # pyright: ignore[reportArgumentType]
    coop.radix_sort_keys(block, keys, descending=1)  # pyright: ignore[reportArgumentType]
    coop.radix_sort_keys(block, keys, 0)  # pyright: ignore[reportCallIssue]
    coop.radix_sort_pairs(block, keys, 1.0)  # pyright: ignore[reportArgumentType]
    coop.radix_sort_pairs(block, float_keys, values)  # pyright: ignore[reportArgumentType]
    coop.radix_sort_pairs(block, keys, complex_values)  # pyright: ignore[reportArgumentType]
    coop.radix_sort_pairs(warp, keys, values)  # pyright: ignore[reportArgumentType]

    # Backend-only routing and representation controls are absent at the root.
    coop.radix_sort_keys(block, keys, launch_metadata={"threads_per_block": 128})  # pyright: ignore[reportCallIssue]
    coop.radix_sort_keys(block, keys, blocked_to_striped=True)  # pyright: ignore[reportCallIssue]
    coop.radix_sort_keys(block, keys, decomposer=object())  # pyright: ignore[reportCallIssue]

    cutlass_block = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_keys = cutlass_coop.ThreadData.from_values(2, 1)
    cutlass_values = cutlass_coop.ThreadData.from_values(np.float32(2), np.float32(1))
    cutlass_complex_values = cutlass_coop.ThreadData.from_values(2j, 1j)
    cutlass_float_keys = cutlass_coop.ThreadData.from_values(2.0, 1.0)
    cutlass_narrow_keys = cutlass_coop.ThreadData.from_values(np.int16(2), np.int16(1))
    cutlass_coop.radix_sort_keys(cutlass_warp, cutlass_keys)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_sort_keys(cutlass_block, cutlass_float_keys)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_sort_keys(cutlass_block, cutlass_narrow_keys)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_sort_keys(cutlass_block, 1.5)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_sort_keys(cutlass_block, cutlass_keys, begin_bit=1.5)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_sort_keys(cutlass_block, cutlass_keys, descending="yes")  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_sort_keys(cutlass_block, cutlass_keys, blocked_to_striped=True)  # pyright: ignore[reportCallIssue]
    cutlass_coop.radix_sort_pairs(cutlass_block, 1.5, np.float32(1))  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_sort_pairs(cutlass_block, cutlass_keys, cutlass_complex_values)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_sort_pairs(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_keys,
        cutlass_values,
        begin_bit=1.5,  # pyright: ignore[reportArgumentType]
    )

    numba_block = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_keys = numba_coop.ThreadData(2, int)
    numba_values = numba_coop.ThreadData(2, np.float32)
    numba_complex_values = numba_coop.ThreadData(2, complex)
    numba_float_keys = numba_coop.ThreadData(2, np.float32)
    numba_coop.radix_sort_keys(numba_warp, numba_keys)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_sort_keys(numba_block, numba_float_keys)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_sort_keys(numba_block, np.float32(1))  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_sort_keys(numba_block, numba_keys, end_bit=opaque)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_sort_keys(numba_block, numba_keys, descending=1)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_sort_keys(numba_block, numba_keys, launch_metadata={})  # pyright: ignore[reportCallIssue]
    numba_coop.radix_sort_pairs(numba_block, numba_keys, np.float32(1))  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_sort_pairs(numba_block, 1, numba_values)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_sort_pairs(numba_block, numba_float_keys, numba_values)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_sort_pairs(numba_block, numba_keys, numba_complex_values)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_sort_pairs(numba_block, numba_keys, numba_values, end_bit=opaque)  # pyright: ignore[reportCallIssue, reportArgumentType]
