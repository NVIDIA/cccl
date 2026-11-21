# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from cuda.tile import _datatype as datatype
from cuda.tile._exception import TileTypeError

from ._base import BlockPrimitive

_SUPPORTED_DTYPES = {
    datatype.bfloat16,
    datatype.float16,
    datatype.float32,
    datatype.float64,
    datatype.int8,
    datatype.uint8,
    datatype.int16,
    datatype.uint16,
    datatype.int32,
    datatype.uint32,
    datatype.int64,
    datatype.uint64,
}


class BlockAdjacentDifferenceSubtractLeft(BlockPrimitive):
    """Check the dtype and item count for left-neighbor subtraction.

    The lowering owns the one-dimensional shape and dispatches a candidate
    width. This adapter checks the per-thread payload accepted by the CUB
    wrapper; it does not read input values or execute the subtraction.
    """

    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
    ) -> None:
        super().__init__(
            name="block_adjacent_difference_subtract_left",
            block_threads=block_threads,
        )
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        self.items_per_thread = items_per_thread
        self.dtype = dtype if isinstance(dtype, datatype.DType) else dtype.dtype
        if self.dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                "coop.adjacent_difference_subtract_left does not support "
                f"dtype {self.dtype}"
            )


class BlockAdjacentDifferenceSubtractRight(BlockPrimitive):
    """Validate the right-neighbor variant with the same payload rules.

    The provider emits its boundary behavior from the selected direction;
    this adapter only checks the compile-time payload description.
    """

    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
    ) -> None:
        super().__init__(
            name="block_adjacent_difference_subtract_right",
            block_threads=block_threads,
        )
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        self.items_per_thread = items_per_thread
        self.dtype = dtype if isinstance(dtype, datatype.DType) else dtype.dtype
        if self.dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                "coop.adjacent_difference_subtract_right does not support "
                f"dtype {self.dtype}"
            )
