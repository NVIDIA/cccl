# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared CUTLASS runtime setup for primitive-family tests."""

import pytest

import cuda.coop.cutlass as coop

from .source import SOURCE_ROOT

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
torch = pytest.importorskip("torch")
typing = pytest.importorskip("cutlass.base_dsl.typing")

from_dlpack = runtime.from_dlpack
Float32 = typing.Float32
Float64 = typing.Float64
Int32 = typing.Int32
Int64 = typing.Int64
Uint8 = typing.Uint8
Uint32 = typing.Uint32
Uint64 = typing.Uint64

FLOAT32_LOWEST = -3.4028234663852886e38

SCAN_SUM_TEMP_STORAGE = coop._block.TempStorage(size_in_bytes=4096, sharing="shared")
ROW_SUM_TEMP_STORAGE = coop._block.TempStorage(size_in_bytes=20, sharing="shared")
ROW_SUM_TEMP_STORAGE_2 = coop._block.TempStorage(size_in_bytes=20, sharing="shared")
SHUFFLE_TEMP_STORAGE = coop._block.TempStorage(size_in_bytes=4096, sharing="shared")
DIFF_DISC_TEMP_STORAGE = coop._block.TempStorage(size_in_bytes=8192, sharing="shared")
EXCHANGE_TEMP_STORAGE = coop._block.TempStorage(size_in_bytes=4096, sharing="shared")
HISTOGRAM_TEMP_STORAGE = coop._block.TempStorage(size_in_bytes=4096, sharing="shared")
RUN_LENGTH_TEMP_STORAGE = coop._block.TempStorage(size_in_bytes=4096, sharing="shared")
RADIX_TEMP_STORAGE = coop._block.TempStorage(size_in_bytes=8192, sharing="shared")
MERGE_TEMP_STORAGE = coop._block.TempStorage(size_in_bytes=8192, sharing="shared")
TOPK_TEMP_STORAGE = coop._block.TempStorage(size_in_bytes=16384, sharing="shared")
GROUP_REDUCE_COMPAT_STORAGE = coop._block.TempStorage(size_in_bytes=1)
TOPK_SCORE_K = 9

LAUNCH_CASES = [
    (16, False),
    (32, False),
    (64, True),
]

LAUNCH_DESCENDING_CASES = [
    (16, False, False),
    (32, False, False),
    (64, True, True),
]


def has_cub_row_reduce_headers() -> bool:
    """Return whether the checked-out CUB supplies RowReduce headers."""

    return (
        SOURCE_ROOT.parents[1] / "cub" / "cub" / "block" / "block_row_reduce.cuh"
    ).is_file()


runtime_pytestmark = [pytest.mark.usefixtures("cutlass_runtime_available")]
