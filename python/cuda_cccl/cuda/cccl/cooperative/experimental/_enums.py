# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import IntEnum, auto
from functools import lru_cache


# Avoid the import of re and subsequent compilation of the regex pattern
# at module import time.
@lru_cache(maxsize=None)
def get_pattern():
    # Match CameCase class names.  Calling `findall()` will return a list of
    # the capitalized words in the given class name.
    import re

    return re.compile(r"[A-Z][^A-Z]*")


def cub_cpp_name(instance):
    cls = instance.__class__
    class_name = cls.__name__
    pattern = get_pattern()
    words = pattern.findall(class_name)
    if words[-1] != "Algorithm":
        raise ValueError(f"Unexpected class name: {class_name}")
    parts = "_".join(word.upper() for word in words[:-1])
    return f"::cub::{parts}_{instance.name}"


class BaseAlgorithmEnum(IntEnum):
    def __str__(self):
        return cub_cpp_name(self)


class BlockLoadAlgorithm(BaseAlgorithmEnum):
    DIRECT = auto()
    STRIPED = auto()
    VECTORIZE = auto()
    TRANSPOSE = auto()
    WARP_TRANSPOSE = auto()
    WARP_TRANSPOSE_TIMESLICED = auto()


class WarpLoadAlgorithm(BaseAlgorithmEnum):
    DIRECT = auto()
    STRIPED = auto()
    VECTORIZE = auto()
    TRANSPOSE = auto()


class BlockStoreAlgorithm(BaseAlgorithmEnum):
    DIRECT = auto()
    STRIPED = auto()
    VECTORIZE = auto()
    TRANSPOSE = auto()
    WARP_TRANSPOSE = auto()
    WARP_TRANSPOSE_TIMESLICED = auto()


class WarpStoreAlgorithm(BaseAlgorithmEnum):
    DIRECT = auto()
    STRIPED = auto()
    VECTORIZE = auto()
    TRANSPOSE = auto()

class BlockHistogramAlgorithm(BaseAlgorithmEnum):
    BLOCK_HISTO_SORT = auto()
    BLOCK_HISTO_ATOMIC = auto()
