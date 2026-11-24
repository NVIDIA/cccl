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


class _BlockExchangeBase(BlockPrimitive):
    """Check the payload dtype and item count for a CUB Exchange adapter.

    Dispatch validates the explicit semantic width first. Public flat-Tile
    permutations use the opposite CUB direction because foreign-call fragments
    have blocked storage; that conversion belongs to the lowering.
    """

    def __init__(
        self,
        *,
        primitive: str,
        name: str,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
    ) -> None:
        super().__init__(name=name, block_threads=block_threads)
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        self.primitive = primitive
        self.items_per_thread = items_per_thread
        self.dtype = dtype if isinstance(dtype, datatype.DType) else dtype.dtype
        if self.dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.{primitive} does not support dtype {self.dtype}"
            )


class BlockExchangeStripedToBlocked(_BlockExchangeBase):
    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
    ) -> None:
        super().__init__(
            primitive="exchange_striped_to_blocked",
            name="block_exchange_striped_to_blocked",
            items_per_thread=items_per_thread,
            dtype=dtype,
            block_threads=block_threads,
        )


class BlockExchangeBlockedToStriped(_BlockExchangeBase):
    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
    ) -> None:
        super().__init__(
            primitive="exchange_blocked_to_striped",
            name="block_exchange_blocked_to_striped",
            items_per_thread=items_per_thread,
            dtype=dtype,
            block_threads=block_threads,
        )
