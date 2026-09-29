# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the common reduction family."""

from typing import Literal, overload

from typing_extensions import TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ReduceAlgorithm,
    ReduceOperator,
    ValidItems,
)

from .thread_group import BlockGroup, ReductionGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)
_ScalarValueT = TypeVar("_ScalarValueT", bound=CommonNumericScalar)

@overload
def reduce(
    group: ReductionGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ItemT:
    """Reduce a full-group payload and return one item of its element type."""

@overload
def reduce(
    group: ReductionGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ScalarValueT:
    """Reduce full-group scalar values while preserving their static type."""

@overload
def reduce(
    group: BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: ValidItems,
    algorithm: ReduceAlgorithm | None = None,
) -> _ScalarValueT:
    """Reduce a valid scalar prefix through direct CUB BlockReduce."""

@overload
def reduce(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> _ItemT:
    """Reduce an array payload with an explicit CUB BlockReduce algorithm."""

@overload
def reduce(
    group: BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> _ScalarValueT:
    """Reduce a scalar with an explicit CUB BlockReduce algorithm."""

@overload
def reduce(
    group: WarpGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: ValidItems,
    algorithm: None = None,
) -> _ScalarValueT:
    """Reduce a valid scalar prefix through direct CUB WarpReduce."""

@overload
def sum(
    group: ReductionGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ItemT:
    """Sum a full-group payload and return one item of its element type."""

@overload
def sum(
    group: ReductionGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ScalarValueT:
    """Sum full-group scalar values while preserving their static type."""

@overload
def sum(
    group: BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: ValidItems,
    algorithm: ReduceAlgorithm | None = None,
) -> _ScalarValueT:
    """Sum a valid scalar prefix through direct CUB BlockReduce."""

@overload
def sum(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> _ItemT:
    """Sum an array payload with an explicit CUB BlockReduce algorithm."""

@overload
def sum(
    group: BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> _ScalarValueT:
    """Sum a scalar with an explicit CUB BlockReduce algorithm."""

@overload
def sum(
    group: WarpGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: ValidItems,
    algorithm: None = None,
) -> _ScalarValueT:
    """Sum a valid scalar prefix through direct CUB WarpReduce."""
