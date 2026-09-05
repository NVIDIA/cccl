# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS reductions."""

from __future__ import annotations

from typing import Any, Literal, overload

from .._typing import ReduceAlgorithm, ReduceOperator, ValidItems
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData
from ._thread_group import BlockGroup, ReductionGroup, WarpGroup
from ._typing import CutlassNumericT, ScalarValueT

@overload
def reduce(
    group: ReductionGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> CutlassNumericT:
    """Reduce a full-group register payload and preserve its element type."""

@overload
def reduce(
    group: ReductionGroup,
    value: ScalarValueT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> ScalarValueT:
    """Reduce full-group CUTLASS scalars while preserving their static type."""

@overload
def reduce(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> CutlassNumericT:
    """Reduce a register payload with an explicit CUB block algorithm."""

@overload
def reduce(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> Any:
    """Reduce a CUTLASS register tensor with an explicit block algorithm."""

@overload
def reduce(
    group: BlockGroup,
    value: ScalarValueT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: ValidItems,
    algorithm: ReduceAlgorithm | None = None,
) -> ScalarValueT:
    """Reduce a scalar through direct CUB BlockReduce at the block root.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def reduce(
    group: BlockGroup,
    value: ScalarValueT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> ScalarValueT:
    """Reduce a scalar with an explicit direct CUB BlockReduce algorithm."""

@overload
def reduce(
    group: WarpGroup,
    value: ScalarValueT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: ValidItems,
    algorithm: None = None,
) -> ScalarValueT:
    """Reduce a valid scalar prefix through direct CUB WarpReduce.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def reduce(
    group: ReductionGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> Any:
    """Reduce a CUTLASS register tensor; its external element type is unknown."""

@overload
def sum(
    group: ReductionGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> CutlassNumericT:
    """Sum a full-group register payload and preserve its element type."""

@overload
def sum(
    group: ReductionGroup,
    value: ScalarValueT,
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> ScalarValueT:
    """Sum full-group CUTLASS scalars while preserving their static type."""

@overload
def sum(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> CutlassNumericT:
    """Sum a register payload with an explicit CUB block algorithm."""

@overload
def sum(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> Any:
    """Sum a CUTLASS register tensor with an explicit block algorithm."""

@overload
def sum(
    group: BlockGroup,
    value: ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: ValidItems,
    algorithm: ReduceAlgorithm | None = None,
) -> ScalarValueT:
    """Sum a scalar through direct CUB BlockReduce at the block root.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def sum(
    group: BlockGroup,
    value: ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> ScalarValueT:
    """Sum a scalar with an explicit direct CUB BlockReduce algorithm."""

@overload
def sum(
    group: WarpGroup,
    value: ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: ValidItems,
    algorithm: None = None,
) -> ScalarValueT:
    """Sum a valid scalar prefix through direct CUB WarpReduce.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def sum(
    group: ReductionGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
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
