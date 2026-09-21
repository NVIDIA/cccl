# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Literal, TypeAlias, overload

import numpy as np
from cutlass import Int32, Int64, Uint32, Uint64
from typing_extensions import TypeVar

from .._core.api.histogram import _Sample
from .._core.api.thread_group import BlockGroup
from .._typing import CommonThreadDataLike, TempStorageLike
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData

_Samples: TypeAlias = (
    CommonThreadDataLike[_Sample] | CutlassTensorSample | CutlassTensorSSASample
)
_Counter = TypeVar(
    "_Counter", np.int32, np.uint32, np.int64, np.uint64, Int32, Uint32, Int64, Uint64
)

@overload
def histogram(
    group: BlockGroup,
    samples: _Samples,
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
    samples: _Samples,
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[_Counter],
    algorithm: Literal["atomic", "sort"] = "atomic",
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[_Counter]: ...
