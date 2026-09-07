# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import IntEnum, auto
from functools import lru_cache


@lru_cache(maxsize=None)
def _get_pattern():
    import re

    # Match CamelCase class names into individual words.
    return re.compile(r"[A-Z][^A-Z]*")


def _cub_cpp_name(instance):
    cls = instance.__class__
    class_name = cls.__name__
    words = _get_pattern().findall(class_name)
    if words[-1] != "Algorithm":
        raise ValueError(f"Unexpected class name: {class_name}")
    parts = "_".join(word.upper() for word in words[:-1])
    return f"::cub::{parts}_{instance.name}"


class BaseAlgorithmEnum(IntEnum):
    """Base enum that renders to the matching CUB C++ algorithm constant."""

    def __str__(self):
        return _cub_cpp_name(self)


class NoAlgorithm(IntEnum):
    """Placeholder algorithm enum for primitives without a CUB algorithm knob."""

    NO_ALGORITHM = auto()


class BlockLoadAlgorithm(BaseAlgorithmEnum):
    """CUB ``BlockLoad`` algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3
    WARP_TRANSPOSE = 4
    WARP_TRANSPOSE_TIMESLICED = 5


class BlockStoreAlgorithm(BaseAlgorithmEnum):
    """CUB ``BlockStore`` algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3
    WARP_TRANSPOSE = 4
    WARP_TRANSPOSE_TIMESLICED = 5


class WarpLoadAlgorithm(BaseAlgorithmEnum):
    """CUB ``WarpLoad`` algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3


class WarpStoreAlgorithm(BaseAlgorithmEnum):
    """CUB ``WarpStore`` algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3


class BlockScanAlgorithm(BaseAlgorithmEnum):
    """CUB ``BlockScan`` algorithm choices."""

    RAKING = 0
    RAKING_MEMOIZE = 1
    WARP_SCANS = 2


class BlockHistogramAlgorithm(IntEnum):
    """CUB ``BlockHistogram`` algorithm choices."""

    SORT = 0
    ATOMIC = 1
