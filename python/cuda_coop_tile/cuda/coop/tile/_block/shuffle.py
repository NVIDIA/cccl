# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from cuda.tile import _datatype as datatype
from cuda.tile._exception import TileTypeError

from ._base import BlockPrimitive

_SUPPORTED_DTYPES = frozenset(
    {
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
)


def _normalize_dtype(dtype):
    if isinstance(dtype, datatype.DType):
        return dtype
    if hasattr(dtype, "dtype"):
        return dtype.dtype
    raise TileTypeError(f"Unsupported dtype {dtype}")


class BlockShuffle(BlockPrimitive):
    """Check the CUB Shuffle overload selected by the lowering.

    Offset and Rotate require one item per thread. Their known distances must
    fit the block width; Rotate also requires a nonnegative distance. Up and
    Down permit multiple items. Runtime distances are checked by generated
    code after this adapter validates a safe placeholder.
    """

    def __init__(
        self,
        *,
        mode: str,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
        distance: int = 1,
    ) -> None:
        super().__init__(name="block_shuffle", block_threads=block_threads)
        if mode not in ("up", "down", "offset", "rotate"):
            raise TileTypeError(f"unsupported shuffle mode {mode}")
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        if mode in ("offset", "rotate") and items_per_thread != 1:
            raise TileTypeError(
                f"coop.shuffle_{mode} requires one item per thread"
            )
        if not isinstance(distance, int):
            raise TileTypeError(f"coop.shuffle_{mode} distance must be an int")
        if mode == "rotate":
            if distance < 0:
                raise TileTypeError("coop.shuffle_rotate distance must be >= 0")
            if distance >= block_threads:
                raise TileTypeError(
                    "coop.shuffle_rotate distance must be < block_threads"
                )
        if mode == "offset" and not (-block_threads < distance < block_threads):
            raise TileTypeError(
                "coop.shuffle_offset distance must satisfy "
                "-block_threads < distance < block_threads"
            )

        self.mode = mode
        self.items_per_thread = items_per_thread
        self.dtype = _normalize_dtype(dtype)
        if self.dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.shuffle_{mode} does not support dtype {self.dtype}"
            )
        self.distance = distance
