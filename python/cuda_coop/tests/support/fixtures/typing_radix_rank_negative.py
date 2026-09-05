# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict negative typing fixture for group-first Radix Rank."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    class RuntimeCompilerInteger:
        """Structural compiler integer that is not a trace-static control."""

        width: int = 32
        signed: bool = True

        @property
        def dtype(self) -> object:
            return object()

        def ir_value(self) -> object:
            return object()

    block = coop.this_block()
    warp = coop.this_warp()
    keys = coop.ThreadData(2, int)
    float_keys = coop.ThreadData(2, float)
    narrow_keys = numba_coop.ThreadData(2, np.int16)
    opaque: object = object()
    runtime_integer = RuntimeCompilerInteger()

    # The common profile is block-only and requires an integral ThreadData.
    coop.radix_rank(block, 1)  # pyright: ignore[reportArgumentType]
    coop.radix_rank(block, np.int32(1))  # pyright: ignore[reportArgumentType]
    coop.radix_rank(block, float_keys)  # pyright: ignore[reportArgumentType]
    coop.radix_rank(block, narrow_keys)  # pyright: ignore[reportArgumentType]
    coop.radix_rank(warp, keys)  # pyright: ignore[reportArgumentType]

    # Bit controls are trace-time Python or NumPy integers, not compiler
    # runtime scalars, floats, or opaque objects.
    coop.radix_rank(block, keys, begin_bit=1.5)  # pyright: ignore[reportArgumentType]
    coop.radix_rank(block, keys, begin_bit=opaque)  # pyright: ignore[reportArgumentType]
    coop.radix_rank(block, keys, begin_bit=runtime_integer)  # pyright: ignore[reportArgumentType]
    coop.radix_rank(block, keys, end_bit=np.float32(8))  # pyright: ignore[reportArgumentType]
    coop.radix_rank(block, keys, radix_bits=opaque)  # pyright: ignore[reportArgumentType]
    coop.radix_rank(block, keys, descending=1)  # pyright: ignore[reportArgumentType]
    coop.radix_rank(block, keys, 0)  # pyright: ignore[reportCallIssue]

    # Qualified-only controls are absent from the portable root.
    coop.radix_rank(block, keys, exclusive_digit_prefix=keys)  # pyright: ignore[reportCallIssue]
    coop.radix_rank(block, keys, launch_metadata={"threads_per_block": 128})  # pyright: ignore[reportCallIssue]

    cutlass_block = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_keys = cutlass_coop.ThreadData.from_values(2, 1)
    cutlass_float_keys = cutlass_coop.ThreadData.from_values(2.0, 1.0)
    cutlass_narrow_keys = cutlass_coop.ThreadData.from_values(np.int16(2), np.int16(1))
    cutlass_float_prefix = cutlass_coop.ThreadData.from_values(0.0, 0.0)
    cutlass_coop.radix_rank(cutlass_warp, cutlass_keys)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_rank(cutlass_block, cutlass_float_keys)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_rank(cutlass_block, cutlass_narrow_keys)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_rank(cutlass_block, 1.5)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_rank(cutlass_block, cutlass_keys, begin_bit=runtime_integer)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_rank(cutlass_block, cutlass_keys, radix_bits=opaque)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_rank(cutlass_block, cutlass_keys, descending="yes")  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.radix_rank(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_keys,
        exclusive_digit_prefix=cutlass_float_prefix,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.radix_rank(cutlass_block, cutlass_keys, temp_storage=None)  # pyright: ignore[reportCallIssue]

    numba_block = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_keys = numba_coop.ThreadData(2, np.uint64)
    numba_float_keys = numba_coop.ThreadData(2, np.float32)
    numba_float_prefix = numba_coop.ThreadData(2, np.float32)
    numba_coop.radix_rank(numba_warp, numba_keys)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_rank(numba_block, numba_float_keys)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_rank(numba_block, np.float32(1))  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_rank(numba_block, numba_keys, end_bit=runtime_integer)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_rank(numba_block, numba_keys, radix_bits=1.0)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_rank(numba_block, numba_keys, descending=1)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.radix_rank(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_keys,
        exclusive_digit_prefix=numba_float_prefix,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.radix_rank(numba_block, numba_keys, launch_metadata={})  # pyright: ignore[reportCallIssue]
