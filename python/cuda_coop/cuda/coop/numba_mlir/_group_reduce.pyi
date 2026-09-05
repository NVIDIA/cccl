# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reduction signatures for supported thread groups."""

from collections.abc import Callable
from typing import Literal, overload

from typing_extensions import TypeVar

from .._typing import (
    ReduceAlgorithm,
    ReduceOperator,
    ScalarValue,
    ThreadDataLike,
    ValidItems,
)
from ._thread_group import BlockGroup, ReductionGroup, WarpGroup

_ItemT = TypeVar("_ItemT")

_ScalarT = TypeVar("_ScalarT", bound=ScalarValue)

@overload
def reduce(
    group: ReductionGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ItemT:
    """Reduce a full-group payload through the CUDAX group provider."""

@overload
def reduce(
    group: ReductionGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ScalarT:
    """Reduce full-group scalar values through the CUDAX group provider."""

@overload
def reduce(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    binary_op: ReduceOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> _ItemT:
    """Reduce ThreadData through explicitly selected CUB BlockReduce."""

@overload
def reduce(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: ReduceOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    broadcast: Literal[False],
    valid_items: ValidItems,
    algorithm: ReduceAlgorithm | None = None,
) -> _ScalarT:
    """Reduce a valid scalar prefix through direct CUB BlockReduce."""

@overload
def reduce(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: ReduceOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> _ScalarT:
    """Reduce a scalar through explicitly selected CUB BlockReduce."""

@overload
def reduce(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: ReduceOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    broadcast: Literal[False],
    valid_items: ValidItems,
    algorithm: None = None,
) -> _ScalarT:
    """Reduce a valid scalar prefix through direct CUB WarpReduce."""

@overload
def reduce(
    group: BlockGroup,
    value: _ItemT,
    /,
    *,
    binary_op: Callable[[_ItemT, _ItemT], _ItemT],
    broadcast: Literal[False],
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
) -> _ItemT:
    """Reduce scalar values with a qualified BlockReduce callback."""

@overload
def reduce(
    group: WarpGroup,
    value: _ItemT,
    /,
    *,
    binary_op: Callable[[_ItemT, _ItemT], _ItemT],
    broadcast: Literal[False],
    valid_items: ValidItems | None = None,
    algorithm: None = None,
) -> _ItemT:
    """Reduce scalar values with a qualified WarpReduce callback."""

@overload
def sum(
    group: ReductionGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ItemT:
    """Sum a full-group payload through the CUDAX group provider."""

@overload
def sum(
    group: ReductionGroup,
    value: _ScalarT,
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ScalarT:
    """Sum full-group scalar values through the CUDAX group provider."""

@overload
def sum(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> _ItemT:
    """Sum ThreadData through explicitly selected CUB BlockReduce."""

@overload
def sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: ValidItems,
    algorithm: ReduceAlgorithm | None = None,
) -> _ScalarT:
    """Sum a valid scalar prefix through direct CUB BlockReduce."""

@overload
def sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> _ScalarT:
    """Sum a scalar through explicitly selected CUB BlockReduce."""

@overload
def sum(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: ValidItems,
    algorithm: None = None,
) -> _ScalarT:
    """Sum a valid scalar prefix through direct CUB WarpReduce."""
