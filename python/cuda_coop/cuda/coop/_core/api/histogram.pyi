# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type histogram counters independently of the sample payload.

Omitting counter_dtype, or spelling it as Python int, gives int32 results. An
explicit supported NumPy counter type determines the result type instead. The
block and input constraints here describe the common API; runtime sample range
and result ownership are explained in the implementation docstring.
"""

from typing import Literal, TypeAlias, overload

import numpy
from typing_extensions import TypeVar

from cuda.coop._typing import (
    CommonThreadDataLike,
    CompilerIntegerLike,
    TempStorageLike,
    ThreadDataLike,
)

from .thread_group import BlockGroup

_Sample: TypeAlias = (
    int
    | numpy.uint8
    | numpy.int32
    | numpy.uint32
    | numpy.int64
    | numpy.uint64
    | CompilerIntegerLike
)
_Counter = TypeVar(
    "_Counter", numpy.int32, numpy.uint32, numpy.int64, numpy.uint64
)

@overload
def histogram(
    group: BlockGroup,
    samples: CommonThreadDataLike[_Sample],
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
    samples: CommonThreadDataLike[_Sample],
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
    samples: CommonThreadDataLike[_Sample],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[_Counter],
    algorithm: Literal["atomic", "sort"] = "atomic",
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_Counter]: ...
