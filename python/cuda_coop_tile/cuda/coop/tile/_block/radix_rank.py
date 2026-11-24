# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from cuda.tile import _datatype as datatype
from cuda.tile._exception import TileTypeError

from ._base import BlockPrimitive
from .radix_sort import _INNER_SCAN_TO_TOKEN, _SMEM_CONFIG_TO_TOKEN

_SUPPORTED_DTYPES = frozenset(
    {
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


class BlockRadixRank(BlockPrimitive):
    """Validate one radix digit interval and its compiled capacity.

    ``radix_bits`` limits the runtime interval width. An omitted end defaults
    to the smaller of the key width and begin plus capacity. Known bounds
    must describe a nonempty interval within both limits. Lowering substitutes
    safe bounds for dynamic values and keeps the real operands separately.
    """

    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
        begin_bit: int = 0,
        end_bit: int | None = None,
        radix_bits: int | None = None,
        descending: bool = False,
        memoize_outer_scan: bool | None = None,
        inner_scan_algorithm: str | None = None,
        smem_config: str | int | None = None,
    ) -> None:
        super().__init__(name="block_radix_rank", block_threads=block_threads)
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        if items_per_thread * block_threads > 65535:
            raise TileTypeError(
                "coop.radix_rank_keys tile size must not exceed 65535"
            )
        if isinstance(begin_bit, bool) or not isinstance(begin_bit, int):
            raise TileTypeError("coop.radix_rank_keys begin_bit must be an int")
        if end_bit is not None and (
            isinstance(end_bit, bool) or not isinstance(end_bit, int)
        ):
            raise TileTypeError("coop.radix_rank_keys end_bit must be an int")
        if radix_bits is not None and (
            isinstance(radix_bits, bool) or not isinstance(radix_bits, int)
        ):
            raise TileTypeError(
                "coop.radix_rank_keys radix_bits must be an int"
            )
        if not isinstance(descending, bool):
            raise TileTypeError(
                "coop.radix_rank_keys descending must be a bool"
            )
        if memoize_outer_scan is not None and not isinstance(
            memoize_outer_scan, bool
        ):
            raise TileTypeError(
                "coop.radix_rank_keys memoize_outer_scan must be a bool"
            )
        if inner_scan_algorithm not in _INNER_SCAN_TO_TOKEN:
            raise TileTypeError(
                "coop.radix_rank_keys inner_scan_algorithm must be one of: "
                "raking, raking_memoize, warp_scans"
            )
        if smem_config not in _SMEM_CONFIG_TO_TOKEN:
            raise TileTypeError(
                "coop.radix_rank_keys smem_config must be one of: default, "
                "four_byte, eight_byte, 4, 8"
            )

        self.items_per_thread = items_per_thread
        self.dtype = _normalize_dtype(dtype)
        if self.dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.radix_rank_keys does not support dtype {self.dtype}"
            )
        self.key_bits = self.dtype.bitwidth
        self.begin_bit = begin_bit
        # RadixBits is a compile-time capacity.  The actual bit interval is a
        # pair of runtime arguments and may be narrower than this capacity.
        self.radix_bits = 4 if radix_bits is None else radix_bits
        self.end_bit = (
            min(self.key_bits, self.begin_bit + self.radix_bits)
            if end_bit is None
            else end_bit
        )
        self.descending = descending
        self.memoize_outer_scan = (
            True if memoize_outer_scan is None else memoize_outer_scan
        )
        self.inner_scan_token = _INNER_SCAN_TO_TOKEN[inner_scan_algorithm]
        self.smem_config_token = _SMEM_CONFIG_TO_TOKEN[smem_config]

        if self.begin_bit < 0:
            raise TileTypeError("coop.radix_rank_keys begin_bit must be >= 0")
        if self.end_bit <= self.begin_bit:
            raise TileTypeError(
                "coop.radix_rank_keys end_bit must be > begin_bit"
            )
        if self.end_bit > self.key_bits:
            raise TileTypeError(
                "coop.radix_rank_keys end_bit exceeds key bit width"
            )
        if self.radix_bits <= 0 or self.radix_bits > 16:
            raise TileTypeError(
                "coop.radix_rank_keys radix_bits must be in the range [1, 16]"
            )
        if self.end_bit - self.begin_bit > self.radix_bits:
            raise TileTypeError(
                "coop.radix_rank_keys end_bit - begin_bit must not exceed "
                "radix_bits"
            )
