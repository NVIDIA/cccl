# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict negative typing fixture for group-first Merge Sort."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    def integer_less(left: int, right: int) -> bool:
        return left < right

    def invalid_comparator(left: str, right: str) -> int:
        return len(left) - len(right)

    block = coop.this_block()
    warp = coop.this_warp()
    thread = coop.this_thread()
    cluster = coop.this_cluster()
    integer_keys = coop.ThreadData(2, int)
    float_keys = coop.ThreadData(2, float)
    numpy_float_keys = coop.ThreadData(2, np.float32)
    narrow_integer_keys = coop.ThreadData(2, np.uint8)
    numeric_values = coop.ThreadData(2, np.float32)
    complex_values = numba_coop.ThreadData(2, complex)
    opaque: object = object()

    # The portable boundary is integral ThreadData only and excludes
    # comparator callbacks.
    coop.merge_sort_keys(block, 1)  # pyright: ignore[reportArgumentType]
    coop.merge_sort_keys(block, [2, 1])  # pyright: ignore[reportArgumentType]
    coop.merge_sort_keys(block, float_keys)  # pyright: ignore[reportArgumentType]
    coop.merge_sort_keys(block, numpy_float_keys)  # pyright: ignore[reportArgumentType]
    coop.merge_sort_keys(block, narrow_integer_keys)  # pyright: ignore[reportArgumentType]
    coop.merge_sort_keys(thread, integer_keys)  # pyright: ignore[reportArgumentType]
    coop.merge_sort_keys(cluster, integer_keys)  # pyright: ignore[reportArgumentType]
    coop.merge_sort_keys(block, integer_keys, compare_op=integer_less)  # pyright: ignore[reportCallIssue]
    coop.merge_sort_keys(block, integer_keys, False)  # pyright: ignore[reportCallIssue]

    # Partial tiles require both controls, an integer count, and a key-typed
    # sentinel.
    coop.merge_sort_keys(block, integer_keys, valid_items=31)  # pyright: ignore[reportArgumentType]
    coop.merge_sort_keys(block, integer_keys, oob_default=0)  # pyright: ignore[reportArgumentType]
    coop.merge_sort_keys(block, integer_keys, valid_items="31", oob_default=0)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.merge_sort_keys(block, integer_keys, valid_items=31.0, oob_default=0)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.merge_sort_keys(block, integer_keys, valid_items=opaque, oob_default=0)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.merge_sort_keys(block, integer_keys, valid_items=31, oob_default=1.5)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.merge_sort_keys(block, integer_keys, descending="yes")  # pyright: ignore[reportArgumentType]
    coop.merge_sort_pairs(block, integer_keys, 1.0)  # pyright: ignore[reportArgumentType]
    coop.merge_sort_pairs(block, float_keys, numeric_values)  # pyright: ignore[reportArgumentType]
    coop.merge_sort_pairs(block, integer_keys, complex_values)  # pyright: ignore[reportArgumentType]
    coop.merge_sort_pairs(block, integer_keys, numeric_values, compare_op=integer_less)  # pyright: ignore[reportCallIssue]

    cutlass_block = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_thread = cutlass_coop.this_thread()
    cutlass_keys = cutlass_coop.ThreadData.from_values(2, 1)
    cutlass_float_keys = cutlass_coop.ThreadData.from_values(2.0, 1.0)
    cutlass_values = cutlass_coop.ThreadData.from_values(np.float32(2), np.float32(1))
    cutlass_complex_values = cutlass_coop.ThreadData.from_values(2j, 1j)
    cutlass_coop.merge_sort_keys(cutlass_block, cutlass_float_keys)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.merge_sort_keys(cutlass_warp, 1)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.merge_sort_keys(cutlass_thread, cutlass_keys)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.merge_sort_keys(cutlass_block, cutlass_keys, compare_op=integer_less)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.merge_sort_keys(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_keys,
        valid_items=31,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.merge_sort_keys(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_keys,
        oob_default=0,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.merge_sort_keys(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_keys,
        valid_items=31,
        oob_default=1.5,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.merge_sort_pairs(cutlass_warp, 1, np.float32(1))  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.merge_sort_pairs(cutlass_block, cutlass_float_keys, cutlass_values)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.merge_sort_pairs(cutlass_block, cutlass_keys, cutlass_complex_values)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.merge_sort_pairs(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_keys,
        cutlass_values,
        valid_items=31,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.merge_sort_pairs(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_keys,
        cutlass_values,
        valid_items=31,
        oob_default=np.float32(0),  # pyright: ignore[reportArgumentType]
    )

    numba_block = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_thread = numba_coop.this_thread()
    numba_keys = numba_coop.ThreadData(2, int)
    numba_values = numba_coop.ThreadData(2, np.float32)
    numba_complex_values = numba_coop.ThreadData(2, complex)
    numba_complex_keys = numba_coop.ThreadData(2, complex)
    numba_coop.merge_sort_keys(numba_block, "one")  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.merge_sort_keys(numba_block, numba_complex_keys)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.merge_sort_keys(  # pyright: ignore[reportCallIssue]
        numba_thread,  # pyright: ignore[reportArgumentType]
        numba_keys,
    )
    numba_coop.merge_sort_keys(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_keys,
        compare_op=invalid_comparator,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.merge_sort_keys(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_keys,
        valid_items=31,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.merge_sort_keys(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_keys,
        oob_default=0,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.merge_sort_keys(numba_block, numba_keys, valid_items=31, oob_default=1.5)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.merge_sort_pairs(numba_block, numba_keys, np.float32(1))  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.merge_sort_pairs(numba_block, 1, numba_values)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.merge_sort_pairs(numba_block, numba_keys, numba_complex_values)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.merge_sort_pairs(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_keys,
        numba_values,
        valid_items=31,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.merge_sort_pairs(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_keys,  # pyright: ignore[reportArgumentType]
        numba_values,
        valid_items=31,
        oob_default=1.5,
    )
