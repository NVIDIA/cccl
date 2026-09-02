# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TypeVar

from cuda.coop import ThreadGroup as ThreadGroup
from cuda.coop._typing import (
    PortableNumericScalar,
    ReduceAlgorithm,
    ReduceOperator,
    ValidItems,
)

_ScalarT = TypeVar("_ScalarT", bound=PortableNumericScalar)

def this_block() -> ThreadGroup: ...
def reduce(
    group: ThreadGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: ReduceOperator | None = ...,
    valid_items: ValidItems | None = ...,
    algorithm: ReduceAlgorithm | None = ...,
) -> _ScalarT: ...
def sum(
    group: ThreadGroup,
    value: _ScalarT,
    /,
    *,
    valid_items: ValidItems | None = ...,
    algorithm: ReduceAlgorithm | None = ...,
) -> _ScalarT: ...
