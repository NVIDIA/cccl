# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Static types for ``cuda.coop.cutlass.reduce_batched``.

The group must be a warp group, and binary_op accepts only built-in operator
names. A common ThreadData input keeps its element type in the result. A CuTe
register tensor or TensorSSA input gives ThreadData[Any], because its element
type is known only when the kernel is traced. The type cannot show which of
the ceil(B / W) result slots hold a batch total.
"""

from typing import Any, Literal, overload

from typing_extensions import TypeVar

from cuda.coop._core.api.thread_group import WarpGroup
from cuda.coop._typing import CommonNumericScalar, CommonThreadDataLike

from ._group_reduce import _BuiltinReduceOperator
from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
)

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
