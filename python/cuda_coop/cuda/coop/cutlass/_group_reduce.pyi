# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS reductions."""

from __future__ import annotations

from typing import Any, Literal, overload

from .._typing import ReduceAlgorithm as _ReduceAlgorithm
from .._typing import ReduceOperator as _ReduceOperator
from .._typing import _ValidItems as _ValidItems
from ._thread_data import ThreadData
from ._thread_data import _CutlassTensorSample as _CutlassTensorSample
from ._thread_data import _CutlassTensorSSASample as _CutlassTensorSSASample
from ._thread_group import _BlockGroup as _BlockGroup
from ._thread_group import _ReductionGroup as _ReductionGroup
from ._thread_group import _WarpGroup as _WarpGroup
from ._typing import _CutlassNumericT as _CutlassNumericT
from ._typing import _ScalarValueT as _ScalarValueT

@overload
def reduce(
    group: _ReductionGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _CutlassNumericT:
    """Reduce a full-group register payload and preserve its element type."""

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
    """Reduce full-group CUTLASS scalars while preserving their static type."""

@overload
def reduce(
    group: _BlockGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: _ReduceAlgorithm,
) -> _CutlassNumericT:
    """Reduce a register payload with an explicit CUB block algorithm."""

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
def reduce(
    group: _ReductionGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> Any:
    """Reduce a CUTLASS register tensor; its external element type is unknown."""

@overload
def sum(
    group: _ReductionGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _CutlassNumericT:
    """Sum a full-group register payload and preserve its element type."""

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
    """Sum full-group CUTLASS scalars while preserving their static type."""

@overload
def sum(
    group: _BlockGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: _ReduceAlgorithm,
) -> _CutlassNumericT:
    """Sum a register payload with an explicit CUB block algorithm."""

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

@overload
def sum(
    group: _ReductionGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> Any:
    """Sum a CUTLASS register tensor; its external element type is unknown."""

__all__ = [
    "reduce",
    "sum",
]
