# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed fresh block histograms."""

from typing import Literal, TypeAlias, overload

import numpy
from typing_extensions import TypeVar

from cuda.coop._typing import (
    CompilerIntegerLike,
    PortableThreadDataLike,
    TempStorageLike,
    ThreadDataLike,
)

from ._thread_group import BlockGroup

_Sample: TypeAlias = (
    int
    | numpy.uint8
    | numpy.int32
    | numpy.uint32
    | numpy.int64
    | numpy.uint64
    | CompilerIntegerLike
)
_Counter = TypeVar("_Counter", numpy.int32, numpy.uint32, numpy.int64, numpy.uint64)

@overload
def histogram(
    group: BlockGroup,
    samples: PortableThreadDataLike[_Sample] | _Sample,
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: None = None,
    algorithm: Literal["atomic", "sort"] = "atomic",
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[numpy.int32]: ...
@overload
def histogram(
    group: BlockGroup,
    samples: PortableThreadDataLike[_Sample] | _Sample,
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[int],
    algorithm: Literal["atomic", "sort"] = "atomic",
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[numpy.int32]: ...
@overload
def histogram(
    group: BlockGroup,
    samples: PortableThreadDataLike[_Sample] | _Sample,
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[_Counter],
    algorithm: Literal["atomic", "sort"] = "atomic",
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_Counter]: ...
