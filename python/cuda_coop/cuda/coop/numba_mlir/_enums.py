# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public load/store algorithm selectors for Numba-CUDA-MLIR."""

from enum import IntEnum
from functools import lru_cache


@lru_cache(maxsize=None)
def _camel_case_pattern():
    import re

    return re.compile(r"[A-Z][^A-Z]*")


class _LoadStoreAlgorithm(IntEnum):
    """Render one enum member as the corresponding CUB constant."""

    def __str__(self) -> str:
        words = _camel_case_pattern().findall(type(self).__name__)
        if not words or words[-1] != "Algorithm":
            raise ValueError(f"Unexpected class name: {type(self).__name__}")
        family = "_".join(word.upper() for word in words[:-1])
        return f"::cub::{family}_{self.name}"


class BlockLoadAlgorithm(_LoadStoreAlgorithm):
    """CUB ``BlockLoad`` algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3
    WARP_TRANSPOSE = 4
    WARP_TRANSPOSE_TIMESLICED = 5


class BlockStoreAlgorithm(_LoadStoreAlgorithm):
    """CUB ``BlockStore`` algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3
    WARP_TRANSPOSE = 4
    WARP_TRANSPOSE_TIMESLICED = 5


class WarpLoadAlgorithm(_LoadStoreAlgorithm):
    """CUB ``WarpLoad`` algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3


class WarpStoreAlgorithm(_LoadStoreAlgorithm):
    """CUB ``WarpStore`` algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3


__all__ = [
    "BlockLoadAlgorithm",
    "BlockStoreAlgorithm",
    "WarpLoadAlgorithm",
    "WarpStoreAlgorithm",
]
