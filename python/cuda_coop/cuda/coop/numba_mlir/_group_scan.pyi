# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scan signatures for block and warp groups."""

from collections.abc import Callable
from typing import Any, Literal, overload

from typing_extensions import TypeVar

from .._typing import ScanAlgorithm as _ScanAlgorithm
from .._typing import ScanOperator as _ScanOperator
from .._typing import ThreadDataLike as _ThreadDataLike
from .._typing import _NonSumScanOperator as _NonSumScanOperator
from .._typing import _ScalarValue as _ScalarValue
from .._typing import _SumScanOperator as _SumScanOperator
from .._typing import _ValidItems as _ValidItems
from ._enums import BlockScanAlgorithm
from ._stateful_function import StatefulFunction
from ._temp_storage import TempStorage
from ._thread_group import _BlockGroup, _WarpGroup

_ItemT = TypeVar("_ItemT")

_ScalarT = TypeVar("_ScalarT", bound=_ScalarValue)

@overload
def scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    initial_value: _ItemT | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return out-of-place block-exclusive sums."""

@overload
def scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator | Callable[[_ItemT, _ItemT], _ItemT],
    initial_value: _ItemT,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return a non-sum block-exclusive scan from an explicit initial value."""

@overload
def scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return an out-of-place block-inclusive scan."""

@overload
def scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: _ScalarT | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-exclusive scalar sum."""

@overload
def scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT],
    initial_value: _ScalarT,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a non-sum block-exclusive scalar scan."""

@overload
def scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-inclusive scalar scan."""

@overload
def scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    prefix_state: _ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive", "inclusive"] = "exclusive",
    scan_op: _ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Scan with an explicit one-item running-prefix state."""

@overload
def scan(
    group: _BlockGroup,
    value: _ScalarT,
    prefix_state: _ThreadDataLike[_ScalarT],
    /,
    *,
    mode: Literal["exclusive", "inclusive"] = "exclusive",
    scan_op: _ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ScalarT:
    """Scan one scalar per thread with an explicit running-prefix state."""

@overload
def scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SumScanOperator | None = None,
    initial_value: _ScalarT | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-exclusive scalar sum."""

@overload
def scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT],
    initial_value: _ScalarT,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a non-sum warp-exclusive scalar scan."""

@overload
def scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-inclusive scalar scan."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return out-of-place block-exclusive sums with the input shape."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-exclusive scalar sum."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    prefix_state: _ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return block-exclusive sums with explicit running-prefix state."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    prefix_state: _ThreadDataLike[_ScalarT],
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ScalarT:
    """Return scalar block-exclusive sums with running-prefix state."""

@overload
def exclusive_sum(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-exclusive scalar sum."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return out-of-place block-inclusive sums with the input shape."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-inclusive scalar sum."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    prefix_state: _ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return block-inclusive sums with explicit running-prefix state."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    prefix_state: _ThreadDataLike[_ScalarT],
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ScalarT:
    """Return scalar block-inclusive sums with running-prefix state."""

@overload
def inclusive_sum(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-inclusive scalar sum."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: _ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    initial_value: _ItemT | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return an out-of-place block-exclusive sum."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: _NonSumScanOperator | Callable[[_ItemT, _ItemT], _ItemT],
    initial_value: _ItemT,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return a non-sum block-exclusive scan."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: _ScalarT | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-exclusive scalar sum."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _NonSumScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT],
    initial_value: _ScalarT,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a non-sum block-exclusive scalar scan."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    prefix_state: _ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: _ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return a block-exclusive scan with explicit running-prefix state."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    prefix_state: _ThreadDataLike[_ScalarT],
    /,
    *,
    scan_op: _ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ScalarT:
    """Return a scalar block-exclusive scan with running-prefix state."""

@overload
def exclusive_scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _SumScanOperator | None = None,
    initial_value: _ScalarT | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-exclusive scalar sum."""

@overload
def exclusive_scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _NonSumScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT],
    initial_value: _ScalarT,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a non-sum warp-exclusive scalar scan."""

@overload
def inclusive_scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: _ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
    prefix_op: Callable[[_ItemT], _ItemT] | None = None,
    block_prefix_callback_op: Callable[[_ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return an out-of-place block-inclusive scan."""

@overload
def inclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
    prefix_op: Callable[[_ScalarT], _ScalarT] | None = None,
    block_prefix_callback_op: Callable[[_ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-inclusive scalar scan."""

@overload
def inclusive_scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    prefix_state: _ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: _ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return a block-inclusive scan with explicit running-prefix state."""

@overload
def inclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    prefix_state: _ThreadDataLike[_ScalarT],
    /,
    *,
    scan_op: _ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: None = None,
    prefix_op: StatefulFunction[Any] | None = None,
    block_prefix_callback_op: StatefulFunction[Any] | None = None,
) -> _ScalarT:
    """Return a scalar block-inclusive scan with running-prefix state."""

@overload
def inclusive_scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-inclusive scalar scan."""
