# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable reduction family."""

from typing import Literal, overload

from typing_extensions import TypeVar

from cuda.coop._typing import (
    PortableNumericScalar,
    ReduceAlgorithm,
    ReduceOperator,
    ThreadDataLike,
    ValidItems,
)

from .thread_group import BlockGroup, ReductionGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=PortableNumericScalar)
_ScalarValueT = TypeVar("_ScalarValueT", bound=PortableNumericScalar)

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
    """Reduce a scalar through direct CUB BlockReduce at the block root.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

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
    """Reduce a scalar with an explicit direct CUB BlockReduce algorithm."""

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
    """Reduce a valid scalar prefix through direct CUB WarpReduce.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

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
    """Sum a scalar through direct CUB BlockReduce at the block root.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

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
    """Sum a scalar with an explicit direct CUB BlockReduce algorithm."""

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
    """Sum a valid scalar prefix through direct CUB WarpReduce.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """
