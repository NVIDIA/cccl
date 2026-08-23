# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict positive typing fixture for group-first Run Length Decode."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import assert_type

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    class CompilerInteger:
        """Structural dynamic compiler integer accepted as a window offset."""

        width: int = 64
        signed: bool = False

        @property
        def dtype(self) -> object:
            return object()

        def ir_value(self) -> object:
            return object()

    block = coop.this_block()
    common_lengths = coop.ThreadData(2, np.uint32)
    assert_type(
        coop.run_length_decode(
            block,
            coop.ThreadData(2, np.uint8),
            common_lengths,
            decoded_items_per_thread=4,
        ),
        coop.ThreadDataLike[np.uint8],
    )
    assert_type(
        coop.run_length_decode(
            block,
            coop.ThreadData(2, np.int64),
            coop.ThreadData(2, np.uint64),
            decoded_items_per_thread=4,
            decoded_window_offset=CompilerInteger(),
        ),
        coop.ThreadDataLike[np.int64],
    )
    assert_type(
        coop.run_length_decode(
            block,
            coop.ThreadData(2, int),
            coop.ThreadData(2, np.int32),
            decoded_items_per_thread=4,
            decoded_window_offset=np.uint64(8),
        ),
        coop.ThreadDataLike[int],
    )

    cutlass_block = cutlass_coop.this_block()
    cutlass_values = cutlass_coop.ThreadData.from_values(
        np.uint8(3),
        np.uint8(7),
    )
    cutlass_lengths = cutlass_coop.ThreadData.from_values(
        np.uint32(1),
        np.uint32(2),
    )
    cutlass_offsets = cutlass_coop.ThreadData(4, np.uint32)
    cutlass_total = cutlass_coop.ThreadData(1, np.uint32)
    assert_type(
        cutlass_coop.run_length_decode(
            cutlass_block,
            cutlass_values,
            cutlass_lengths,
            decoded_items_per_thread=4,
            decoded_window_offset=CompilerInteger(),
            relative_offsets=cutlass_offsets,
            total_decoded_size=cutlass_total,
            decoded_offset_dtype=np.uint32,
        ),
        cutlass_coop.ThreadData[np.uint8],
    )
    assert_type(
        cutlass_coop.run_length_decode(
            cutlass_block,
            np.int64(9),
            np.uint64(3),
            decoded_items_per_thread=2,
        ),
        cutlass_coop.ThreadData[np.int64],
    )

    numba_block = numba_coop.this_block()
    numba_values = numba_coop.ThreadData(2, np.int32)
    numba_lengths = numba_coop.ThreadData(2, np.int64)
    numba_offsets = numba_coop.ThreadData(4, np.int64)
    numba_total = numba_coop.ThreadData(1, np.int64)
    assert_type(
        numba_coop.run_length_decode(
            numba_block,
            numba_values,
            numba_lengths,
            decoded_items_per_thread=4,
            decoded_window_offset=np.int64(3),
            relative_offsets=numba_offsets,
            total_decoded_size=numba_total,
            decoded_offset_dtype=np.int64,
        ),
        coop.ThreadDataLike[np.int32],
    )
    assert_type(
        numba_coop.run_length_decode(
            numba_block,
            numba_coop.ThreadData(2, np.float32),
            numba_coop.ThreadData(2, np.uint16),
            decoded_items_per_thread=np.int32(4),
        ),
        coop.ThreadDataLike[np.float32],
    )

    # Qualified groups and payloads also satisfy the portable root contract.
    assert_type(
        coop.run_length_decode(
            cutlass_block,
            cutlass_values,
            cutlass_lengths,
            decoded_items_per_thread=4,
        ),
        coop.ThreadDataLike[np.uint8],
    )
    assert_type(
        coop.run_length_decode(
            numba_block,
            numba_values,
            numba_lengths,
            decoded_items_per_thread=4,
        ),
        coop.ThreadDataLike[np.int32],
    )
