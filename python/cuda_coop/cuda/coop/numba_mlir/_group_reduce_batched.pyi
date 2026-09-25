# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Literal

from typing_extensions import TypeVar

from cuda.coop._typing import (
    PortableNumericScalar,
    PortableThreadDataLike,
    ReduceOperator,
    ThreadDataLike,
)

from ._thread_group import WarpGroup

_ItemT = TypeVar("_ItemT", bound=PortableNumericScalar)

def reduce_batched(
    group: WarpGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    binary_op: ReduceOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    output_layout: Literal["striped", "blocked"] = "striped",
) -> ThreadDataLike[_ItemT]: ...
