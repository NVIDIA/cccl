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


class BlockDiscontinuityFlagHeads(BlockPrimitive):
    """Validate the payload used to mark unequal left-neighbor pairs.

    The lowering creates int32 flags with the input shape. This adapter checks
    the supported input dtype and positive per-thread count before emission.
    """

    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
    ) -> None:
        super().__init__(
            name="block_discontinuity_flag_heads", block_threads=block_threads
        )
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        self.items_per_thread = items_per_thread
        self.dtype = dtype if isinstance(dtype, datatype.DType) else dtype.dtype
        if self.dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.discontinuity_flag_heads does not support "
                f"dtype {self.dtype}"
            )


class BlockDiscontinuityFlagTails(BlockPrimitive):
    """Validate the payload used to mark unequal right-neighbor pairs."""

    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
    ) -> None:
        super().__init__(
            name="block_discontinuity_flag_tails", block_threads=block_threads
        )
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        self.items_per_thread = items_per_thread
        self.dtype = dtype if isinstance(dtype, datatype.DType) else dtype.dtype
        if self.dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.discontinuity_flag_tails does not support "
                f"dtype {self.dtype}"
            )


class BlockDiscontinuityFlagHeadsAndTails(BlockPrimitive):
    """Validate one payload for the combined head/tail provider call.

    Both output flag Tiles use int32 elements. The combined call shares its
    input and CUB scratch instead of emitting two independent primitives.
    """

    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
    ) -> None:
        super().__init__(
            name="block_discontinuity_flag_heads_and_tails",
            block_threads=block_threads,
        )
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        self.items_per_thread = items_per_thread
        self.dtype = dtype if isinstance(dtype, datatype.DType) else dtype.dtype
        if self.dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                "coop.discontinuity_flag_heads_and_tails does not support "
                f"dtype {self.dtype}"
            )
