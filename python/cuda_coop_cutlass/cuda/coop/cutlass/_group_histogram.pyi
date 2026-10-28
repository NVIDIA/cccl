# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declare static result types for ``cuda.coop.cutlass.histogram``.

The counter type does not depend on the sample type. An omitted selector or
Python int gives Int32 counters. An explicit NumPy or CuTe selector sets the
counter type. Result extent follows bins_per_thread, not the sample count.
"""

from typing import Literal, TypeAlias, overload

import numpy as np
from typing_extensions import TypeVar

from cutlass import Int32, Int64, Uint32, Uint64

from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
)
from ._typing import (
    BlockGroup,
    CommonNumericScalar,
    CommonThreadDataLike,
    TempStorageLike,
)

_Sample = TypeVar("_Sample", bound=CommonNumericScalar)

_Samples: TypeAlias = (
    CommonThreadDataLike[_Sample] | CutlassTensorSample | CutlassTensorSSASample
)
_Counter = TypeVar(
    "_Counter",
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    Int32,
    Uint32,
    Int64,
    Uint64,
)
_NumpyCounter = TypeVar(
    "_NumpyCounter", np.int32, np.uint32, np.int64, np.uint64
)

@overload
def histogram(
    group: BlockGroup,
    samples: _Samples[_Sample],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: None | type[int] = None,
    algorithm: Literal["atomic", "sort"] = "atomic",
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[Int32]: ...
@overload
def histogram(
    group: BlockGroup,
    samples: _Samples[_Sample],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: np.dtype[_NumpyCounter],
    algorithm: Literal["atomic", "sort"] = "atomic",
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[_NumpyCounter]: ...
@overload
def histogram(
    group: BlockGroup,
    samples: _Samples[_Sample],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[_Counter],
    algorithm: Literal["atomic", "sort"] = "atomic",
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[_Counter]: ...
