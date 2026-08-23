# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict negative typing fixture for group-first TopK."""

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
    narrow_keys = coop.ThreadData(2, np.uint8)
    values = coop.ThreadData(2, np.float32)
    complex_values = numba_coop.ThreadData(2, complex)
    opaque: object = object()

    # The common profile is one-dimensional block-only and restricted to
    # portable integer-key and numeric-value ThreadData payloads.
    coop.topk_max_keys(block, 1, 7)  # pyright: ignore[reportArgumentType]
    coop.topk_min_keys(block, float_keys, 7)  # pyright: ignore[reportArgumentType]
    coop.topk_max_keys(block, narrow_keys, 7)  # pyright: ignore[reportArgumentType]
    coop.topk_min_keys(warp, keys, 7)  # pyright: ignore[reportArgumentType]
    coop.topk_max_pairs(block, keys, 1.0, 7)  # pyright: ignore[reportArgumentType]
    coop.topk_min_pairs(block, float_keys, values, 7)  # pyright: ignore[reportArgumentType]
    coop.topk_max_pairs(block, keys, complex_values, 7)  # pyright: ignore[reportArgumentType]
    coop.topk_min_pairs(warp, keys, values, 7)  # pyright: ignore[reportArgumentType]

    # Counts and bit bounds are Python, NumPy, or structural compiler integers.
    coop.topk_max_keys(block, keys, 7.0)  # pyright: ignore[reportArgumentType]
    coop.topk_max_keys(block, keys, opaque)  # pyright: ignore[reportArgumentType]
    coop.topk_min_keys(block, keys, 7, valid_items=31.0)  # pyright: ignore[reportArgumentType]
    coop.topk_min_keys(block, keys, 7, valid_items=opaque)  # pyright: ignore[reportArgumentType]
    coop.topk_max_keys(block, keys, 7, begin_bit=np.float32(0))  # pyright: ignore[reportArgumentType]
    coop.topk_max_keys(block, keys, 7, end_bit=opaque)  # pyright: ignore[reportArgumentType]
    coop.topk_min_keys(block, keys, 7, launch_metadata={})  # pyright: ignore[reportCallIssue]
    coop.topk_min_keys(block, keys, 7, 31)  # pyright: ignore[reportCallIssue]

    cutlass_block = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_keys = cutlass_coop.ThreadData.from_values(3, 1)
    cutlass_values = cutlass_coop.ThreadData.from_values(np.float32(30), np.float32(10))
    cutlass_complex = cutlass_coop.ThreadData.from_values(3 + 1j, 1 + 0j)
    cutlass_narrow = cutlass_coop.ThreadData.from_values(np.int16(3), np.int16(1))
    cutlass_coop.topk_max_keys(cutlass_warp, cutlass_keys, 7)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.topk_min_keys(cutlass_block, cutlass_complex, 7)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.topk_max_keys(cutlass_block, cutlass_narrow, 7)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.topk_min_keys(cutlass_block, cutlass_keys, "7")  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.topk_max_keys(cutlass_block, cutlass_keys, 7, valid_items=31.0)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.topk_min_pairs(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_keys,
        cutlass_values,
        7,
        begin_bit=opaque,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.topk_max_pairs(cutlass_block, cutlass_keys, np.float32(1), 7)  # pyright: ignore[reportCallIssue, reportArgumentType]

    numba_block = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_keys = numba_coop.ThreadData(2, np.int32)
    numba_values = numba_coop.ThreadData(2, np.float32)
    numba_complex = numba_coop.ThreadData(2, complex)
    numba_coop.topk_max_keys(numba_warp, numba_keys, 7)  # pyright: ignore[reportArgumentType]
    numba_coop.topk_min_keys(numba_block, np.int32(3), 7)  # pyright: ignore[reportArgumentType]
    numba_coop.topk_max_keys(numba_block, numba_complex, 7)  # pyright: ignore[reportArgumentType]
    numba_coop.topk_min_keys(numba_block, numba_keys, np.float32(7))  # pyright: ignore[reportArgumentType]
    numba_coop.topk_max_pairs(numba_block, numba_keys, numba_values, 7, end_bit=opaque)  # pyright: ignore[reportArgumentType]
    numba_coop.topk_min_pairs(
        numba_block,
        numba_keys,
        numba_values,
        7,
        launch_metadata={},  # pyright: ignore[reportCallIssue]
    )
