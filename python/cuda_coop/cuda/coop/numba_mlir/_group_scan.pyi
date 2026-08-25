# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scan signatures for block and warp groups."""

from collections.abc import Callable
from typing import Any, Literal, overload

from typing_extensions import TypeVar

from .._typing import (
    NonSumScanOperator,
    ScalarValue,
    ScanAlgorithm,
    ScanOperator,
    SumScanOperator,
    ThreadDataLike,
    ValidItems,
)
from ._enums import BlockScanAlgorithm
from ._stateful_function import StatefulFunction
from ._temp_storage import TempStorage
from ._thread_group import BlockGroup, WarpGroup

_ItemT = TypeVar("_ItemT")

_ScalarT = TypeVar("_ScalarT", bound=ScalarValue)

@overload
def scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    initial_value: _ItemT | None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> ThreadDataLike[_ItemT]:
    """Return out-of-place block-exclusive sums."""

@overload
def scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: NonSumScanOperator | Callable[[_ItemT, _ItemT], _ItemT],
    initial_value: _ItemT,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> ThreadDataLike[_ItemT]:
    """Return a non-sum block-exclusive scan from an explicit initial value."""

@overload
def scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> ThreadDataLike[_ItemT]:
    """Return an out-of-place block-inclusive scan."""

@overload
def scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: _ScalarT | None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-exclusive scalar sum."""

@overload
def scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: NonSumScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT],
    initial_value: _ScalarT,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a non-sum block-exclusive scalar scan."""

@overload
def scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-inclusive scalar scan."""

@overload
def scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    prefix_state: ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive", "inclusive"] = "exclusive",
    scan_op: ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> ThreadDataLike[_ItemT]:
    """Scan with an explicit one-item running-prefix state."""

@overload
def scan(
    group: BlockGroup,
    value: _ScalarT,
    prefix_state: ThreadDataLike[_ScalarT],
    /,
    *,
    mode: Literal["exclusive", "inclusive"] = "exclusive",
    scan_op: ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ScalarT:
    """Scan one scalar per thread with an explicit running-prefix state."""

@overload
def scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: _ScalarT | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-exclusive scalar sum."""

@overload
def scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: NonSumScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT],
    initial_value: _ScalarT,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a non-sum warp-exclusive scalar scan."""

@overload
def scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-inclusive scalar scan."""

@overload
def exclusive_sum(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> ThreadDataLike[_ItemT]:
    """Return out-of-place block-exclusive sums with the input shape."""

@overload
def exclusive_sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-exclusive scalar sum."""

@overload
def exclusive_sum(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    prefix_state: ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> ThreadDataLike[_ItemT]:
    """Return block-exclusive sums with explicit running-prefix state."""

@overload
def exclusive_sum(
    group: BlockGroup,
    value: _ScalarT,
    prefix_state: ThreadDataLike[_ScalarT],
    /,
    *,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ScalarT:
    """Return scalar block-exclusive sums with running-prefix state."""

@overload
def exclusive_sum(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-exclusive scalar sum."""

@overload
def inclusive_sum(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> ThreadDataLike[_ItemT]:
    """Return out-of-place block-inclusive sums with the input shape."""

@overload
def inclusive_sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-inclusive scalar sum."""

@overload
def inclusive_sum(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    prefix_state: ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> ThreadDataLike[_ItemT]:
    """Return block-inclusive sums with explicit running-prefix state."""

@overload
def inclusive_sum(
    group: BlockGroup,
    value: _ScalarT,
    prefix_state: ThreadDataLike[_ScalarT],
    /,
    *,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ScalarT:
    """Return scalar block-inclusive sums with running-prefix state."""

@overload
def inclusive_sum(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-inclusive scalar sum."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    initial_value: _ItemT | None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> ThreadDataLike[_ItemT]:
    """Return an out-of-place block-exclusive sum."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: NonSumScanOperator | Callable[[_ItemT, _ItemT], _ItemT],
    initial_value: _ItemT,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> ThreadDataLike[_ItemT]:
    """Return a non-sum block-exclusive scan."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: _ScalarT | None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-exclusive scalar sum."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: NonSumScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT],
    initial_value: _ScalarT,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a non-sum block-exclusive scalar scan."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    prefix_state: ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> ThreadDataLike[_ItemT]:
    """Return a block-exclusive scan with explicit running-prefix state."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    prefix_state: ThreadDataLike[_ScalarT],
    /,
    *,
    scan_op: ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ScalarT:
    """Return a scalar block-exclusive scan with running-prefix state."""

@overload
def exclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: _ScalarT | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-exclusive scalar sum."""

@overload
def exclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: NonSumScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT],
    initial_value: _ScalarT,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a non-sum warp-exclusive scalar scan."""

@overload
def inclusive_scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> ThreadDataLike[_ItemT]:
    """Return an out-of-place block-inclusive scan."""

@overload
def inclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-inclusive scalar scan."""

@overload
def inclusive_scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    prefix_state: ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> ThreadDataLike[_ItemT]:
    """Return a block-inclusive scan with explicit running-prefix state."""

@overload
def inclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    prefix_state: ThreadDataLike[_ScalarT],
    /,
    *,
    scan_op: ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ScalarT:
    """Return a scalar block-inclusive scan with running-prefix state."""

@overload
def inclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-inclusive scalar scan."""
