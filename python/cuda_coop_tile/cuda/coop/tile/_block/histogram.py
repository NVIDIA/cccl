# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from cuda.tile._exception import TileTypeError

from ._base import BlockPrimitive


class BlockHistogram(BlockPrimitive):
    """Validate per-thread sample and output-bin counts for uint8 input.

    Bins must fit the uint8 domain and divide evenly across the candidate
    block width. The lowering checks input dtype and shape; the native
    provider admits the final physical configuration independently.
    """

    def __init__(
        self,
        *,
        block_threads: int = 128,
        items_per_thread: int = 4,
        bins: int = 256,
    ) -> None:
        super().__init__(
            name=f"block_histogram_{bins}", block_threads=block_threads
        )
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        if bins <= 0:
            raise TileTypeError("bins must be > 0")
        if bins > 256:
            raise TileTypeError("bins must be <= 256 for uint8 samples")
        if bins % block_threads != 0:
            raise TileTypeError("bins must be divisible by block_threads")
        self.items_per_thread = items_per_thread
        self.bins = bins


class BlockHistogram256(BlockHistogram):
    def __init__(
        self,
        *,
        block_threads: int = 128,
        items_per_thread: int = 4,
        bins: int = 256,
    ) -> None:
        super().__init__(
            block_threads=block_threads,
            items_per_thread=items_per_thread,
            bins=bins,
        )
