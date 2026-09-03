# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public Block Load/Store algorithm selectors for Numba-CUDA-MLIR."""

from enum import IntEnum


class _LoadStoreAlgorithm(IntEnum):
    """Render one enum member as the corresponding CUB constant."""

    def __str__(self) -> str:
        family = type(self).__name__.removesuffix("Algorithm")
        words = []
        start = 0
        for index, character in enumerate(family):
            if index and character.isupper():
                words.append(family[start:index])
                start = index
        words.append(family[start:])
        return f"::cub::{'_'.join(word.upper() for word in words)}_{self.name}"


class BlockLoadAlgorithm(_LoadStoreAlgorithm):
    """CUB BlockLoad algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3
    WARP_TRANSPOSE = 4
    WARP_TRANSPOSE_TIMESLICED = 5


class BlockStoreAlgorithm(_LoadStoreAlgorithm):
    """CUB BlockStore algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3
    WARP_TRANSPOSE = 4
    WARP_TRANSPOSE_TIMESLICED = 5


__all__ = ["BlockLoadAlgorithm", "BlockStoreAlgorithm"]
