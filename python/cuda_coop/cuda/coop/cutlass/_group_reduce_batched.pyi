# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Literal, overload

from typing_extensions import TypeVar

from cuda.coop._core.api.thread_group import WarpGroup
from cuda.coop._typing import CommonNumericScalar, CommonThreadDataLike

from ._group_reduce import _BuiltinReduceOperator
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)

@overload
def reduce_batched(
    group: WarpGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    binary_op: _BuiltinReduceOperator | None = None,
    output_layout: Literal["striped", "blocked"] = "striped",
) -> ThreadData[_ItemT]: ...
@overload
def reduce_batched(
    group: WarpGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    binary_op: _BuiltinReduceOperator | None = None,
    output_layout: Literal["striped", "blocked"] = "striped",
) -> ThreadData[Any]: ...
