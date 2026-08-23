# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable reduction family."""

from typing import Literal, overload

from typing_extensions import TypeVar

from cuda.coop._typing import ReduceAlgorithm as _ReduceAlgorithm
from cuda.coop._typing import ReduceOperator as _ReduceOperator
from cuda.coop._typing import ThreadDataLike as ThreadDataLike
from cuda.coop._typing import _PortableNumericScalar as _PortableNumericScalar
from cuda.coop._typing import _ValidItems as _ValidItems

from .thread_group import _BlockGroup, _ReductionGroup, _WarpGroup

_ItemT = TypeVar("_ItemT", bound=_PortableNumericScalar)
_ScalarValueT = TypeVar("_ScalarValueT", bound=_PortableNumericScalar)

@overload
def reduce(
    group: _ReductionGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ItemT:
    """Reduce a full-group payload and return one item of its element type."""

@overload
def reduce(
    group: _ReductionGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ScalarValueT:
    """Reduce full-group scalar values while preserving their static type."""

@overload
def reduce(
    group: _BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: _ReduceAlgorithm | None = None,
) -> _ScalarValueT:
    """Reduce a scalar through direct CUB BlockReduce at the block root.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def reduce(
    group: _BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: _ReduceAlgorithm,
) -> _ScalarValueT:
    """Reduce a scalar with an explicit direct CUB BlockReduce algorithm."""

@overload
def reduce(
    group: _WarpGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: None = None,
) -> _ScalarValueT:
    """Reduce a valid scalar prefix through direct CUB WarpReduce.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def sum(
    group: _ReductionGroup,
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
    group: _ReductionGroup,
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
    group: _BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: _ReduceAlgorithm | None = None,
) -> _ScalarValueT:
    """Sum a scalar through direct CUB BlockReduce at the block root.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def sum(
    group: _BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: _ReduceAlgorithm,
) -> _ScalarValueT:
    """Sum a scalar with an explicit direct CUB BlockReduce algorithm."""

@overload
def sum(
    group: _WarpGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: None = None,
) -> _ScalarValueT:
    """Sum a valid scalar prefix through direct CUB WarpReduce.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """
