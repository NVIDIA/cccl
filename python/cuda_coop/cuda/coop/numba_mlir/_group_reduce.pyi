# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar block and physical-warp reduction signatures."""

from typing import Literal, TypeVar, overload

from cuda.coop._typing import (
    PortableNumericScalar,
    ReduceAlgorithm,
    ReduceOperator,
    ValidItems,
)

from ._thread_group import ThreadGroup

_ScalarT = TypeVar("_ScalarT", bound=PortableNumericScalar)

@overload
def reduce(
    group: ThreadGroup[Literal["block"]],
    value: _ScalarT,
    /,
    *,
    binary_op: ReduceOperator | None = ...,
    valid_items: ValidItems | None = ...,
    algorithm: ReduceAlgorithm | None = ...,
) -> _ScalarT: ...
@overload
def reduce(
    group: ThreadGroup[Literal["warp"]],
    value: _ScalarT,
    /,
    *,
    binary_op: ReduceOperator | None = ...,
    valid_items: ValidItems | None = ...,
    algorithm: None = ...,
) -> _ScalarT: ...
@overload
def sum(
    group: ThreadGroup[Literal["block"]],
    value: _ScalarT,
    /,
    *,
    valid_items: ValidItems | None = ...,
    algorithm: ReduceAlgorithm | None = ...,
) -> _ScalarT: ...
@overload
def sum(
    group: ThreadGroup[Literal["warp"]],
    value: _ScalarT,
    /,
    *,
    valid_items: ValidItems | None = ...,
    algorithm: None = ...,
) -> _ScalarT: ...

__all__ = ["reduce", "sum"]
