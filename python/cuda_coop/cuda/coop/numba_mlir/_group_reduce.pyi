# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar BlockReduce signatures for the qualified backend."""

from typing import TypeVar

from cuda.coop._typing import (
    PortableNumericScalar,
    ReduceAlgorithm,
    ReduceOperator,
    ValidItems,
)

from ._thread_group import ThreadGroup

_ScalarT = TypeVar("_ScalarT", bound=PortableNumericScalar)

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

__all__ = ["reduce", "sum"]
