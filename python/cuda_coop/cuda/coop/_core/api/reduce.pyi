# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TypeVar, overload

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ReduceAlgorithm,
    ReduceOperator,
    TempStorageLike,
    ValidItems,
)

from .thread_group import BlockGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)

@overload
def reduce(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    binary_op: ReduceOperator | None = None,
    valid_items: None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def reduce(
    group: BlockGroup,
    value: _ItemT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def reduce(
    group: WarpGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    binary_op: ReduceOperator | None = None,
    valid_items: None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def reduce(
    group: WarpGroup,
    value: _ItemT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    valid_items: ValidItems | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def sum(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    valid_items: None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def sum(
    group: BlockGroup,
    value: _ItemT,
    /,
    *,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def sum(
    group: WarpGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    valid_items: None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def sum(
    group: WarpGroup,
    value: _ItemT,
    /,
    *,
    valid_items: ValidItems | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""
