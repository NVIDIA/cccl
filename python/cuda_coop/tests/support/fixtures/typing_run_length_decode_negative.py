# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict negative typing fixture for group-first Run Length Decode."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    block = coop.this_block()
    warp = coop.this_warp()
    values = coop.ThreadData(2, np.uint8)
    lengths = coop.ThreadData(2, np.uint32)
    float_values = coop.ThreadData(2, np.float32)
    narrow_values = numba_coop.ThreadData(2, np.int16)
    byte_lengths = coop.ThreadData(2, np.uint8)
    float_lengths = coop.ThreadData(2, np.float32)
    opaque: object = object()

    # The common profile is block-only and requires matching integral
    # ThreadData inputs from its certified dtype intersection.
    coop.run_length_decode(block, 1, lengths, decoded_items_per_thread=2)  # pyright: ignore[reportArgumentType]
    coop.run_length_decode(block, values, 1, decoded_items_per_thread=2)  # pyright: ignore[reportArgumentType]
    coop.run_length_decode(block, float_values, lengths, decoded_items_per_thread=2)  # pyright: ignore[reportArgumentType]
    coop.run_length_decode(block, narrow_values, lengths, decoded_items_per_thread=2)  # pyright: ignore[reportArgumentType]
    coop.run_length_decode(block, values, byte_lengths, decoded_items_per_thread=2)  # pyright: ignore[reportArgumentType]
    coop.run_length_decode(block, values, float_lengths, decoded_items_per_thread=2)  # pyright: ignore[reportArgumentType]
    coop.run_length_decode(warp, values, lengths, decoded_items_per_thread=2)  # pyright: ignore[reportArgumentType]

    # The output extent is keyword-only and trace-static; the window offset is
    # an integer value rather than a float or opaque object.
    coop.run_length_decode(block, values, lengths, 2)  # pyright: ignore[reportCallIssue]
    coop.run_length_decode(block, values, lengths, decoded_items_per_thread=2.0)  # pyright: ignore[reportArgumentType]
    coop.run_length_decode(
        block,
        values,
        lengths,
        decoded_items_per_thread=2,
        decoded_window_offset=1.5,  # pyright: ignore[reportArgumentType]
    )
    coop.run_length_decode(
        block,
        values,
        lengths,
        decoded_items_per_thread=2,
        decoded_window_offset=opaque,  # pyright: ignore[reportArgumentType]
    )
    coop.run_length_decode(
        block,
        values,
        lengths,
        decoded_items_per_thread=2,
        relative_offsets=lengths,  # pyright: ignore[reportCallIssue]
    )
    coop.run_length_decode(
        block,
        values,
        lengths,
        decoded_items_per_thread=2,
        total_decoded_size=lengths,  # pyright: ignore[reportCallIssue]
    )
    coop.run_length_decode(
        block,
        values,
        lengths,
        decoded_items_per_thread=2,
        temp_storage=None,  # pyright: ignore[reportCallIssue]
    )

    cutlass_block = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_values = cutlass_coop.ThreadData.from_values(np.uint8(1), np.uint8(2))
    cutlass_lengths = cutlass_coop.ThreadData.from_values(np.uint32(1), np.uint32(2))
    cutlass_complex_values = cutlass_coop.ThreadData.from_values(1 + 0j, 2 + 0j)
    cutlass_narrow_values = cutlass_coop.ThreadData.from_values(
        np.int16(1), np.int16(2)
    )
    cutlass_float_offsets = cutlass_coop.ThreadData(2, np.float32)
    cutlass_signed_offsets = cutlass_coop.ThreadData(2, np.int32)
    cutlass_signed_total = cutlass_coop.ThreadData(1, np.int32)
    cutlass_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        cutlass_warp,  # pyright: ignore[reportArgumentType]
        cutlass_values,
        cutlass_lengths,
        decoded_items_per_thread=2,
    )
    cutlass_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        1.5,  # pyright: ignore[reportArgumentType]
        np.uint32(1),
        decoded_items_per_thread=2,
    )
    cutlass_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_complex_values,  # pyright: ignore[reportArgumentType]
        cutlass_lengths,
        decoded_items_per_thread=2,
    )
    cutlass_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_narrow_values,  # pyright: ignore[reportArgumentType]
        cutlass_lengths,
        decoded_items_per_thread=2,
    )
    cutlass_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_values,
        np.uint32(1),  # pyright: ignore[reportArgumentType]
        decoded_items_per_thread=2,
    )
    cutlass_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        np.uint8(1),
        cutlass_lengths,  # pyright: ignore[reportArgumentType]
        decoded_items_per_thread=2,
    )
    cutlass_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_values,
        cutlass_lengths,
        decoded_items_per_thread=2,
        relative_offsets=cutlass_float_offsets,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_values,
        cutlass_lengths,
        decoded_items_per_thread=2,
        relative_offsets=cutlass_signed_offsets,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_values,
        cutlass_lengths,
        decoded_items_per_thread=2,
        total_decoded_size=cutlass_signed_total,  # pyright: ignore[reportArgumentType]
    )

    numba_block = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_values = numba_coop.ThreadData(2, np.int32)
    numba_lengths = numba_coop.ThreadData(2, np.uint64)
    numba_complex_values = numba_coop.ThreadData(2, complex)
    numba_signed_offsets = numba_coop.ThreadData(2, np.int64)
    numba_signed_total = numba_coop.ThreadData(1, np.int64)
    numba_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        numba_warp,  # pyright: ignore[reportArgumentType]
        numba_values,
        numba_lengths,
        decoded_items_per_thread=2,
    )
    numba_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        numba_block,
        1,  # pyright: ignore[reportArgumentType]
        numba_lengths,
        decoded_items_per_thread=2,
    )
    numba_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_complex_values,  # pyright: ignore[reportArgumentType]
        numba_lengths,
        decoded_items_per_thread=2,
    )
    numba_coop.run_length_decode(
        numba_block,
        numba_values,
        numba_lengths,
        decoded_items_per_thread=2,
        launch_metadata={},  # pyright: ignore[reportCallIssue]
    )
    numba_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_values,
        numba_lengths,
        decoded_items_per_thread=2,
        relative_offsets=numba_signed_offsets,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.run_length_decode(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_values,
        numba_lengths,
        decoded_items_per_thread=2,
        total_decoded_size=numba_signed_total,  # pyright: ignore[reportArgumentType]
    )
