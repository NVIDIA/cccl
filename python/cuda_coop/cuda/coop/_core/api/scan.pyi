# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable scan family."""

from typing import Literal, overload

from typing_extensions import TypeVar

from cuda.coop._typing import (
    ContextualInitialValue,
    NonSumScanOperator,
    PortableNumericScalar,
    PortableThreadDataLike,
    ScanAlgorithm,
    ScanOperator,
    SumScanOperator,
    TempStorageLike,
    ThreadDataLike,
)

from .thread_group import BlockGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=PortableNumericScalar)
_ScalarT = TypeVar("_ScalarT", bound=PortableNumericScalar)

@overload
def scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: ContextualInitialValue[_ScalarT] | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT: ...
@overload
def scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: NonSumScanOperator,
    initial_value: ContextualInitialValue[_ScalarT],
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT: ...
@overload
def scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: ScanOperator | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT: ...
@overload
def scan(
    group: BlockGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: ContextualInitialValue[_ItemT] | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def scan(
    group: BlockGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: NonSumScanOperator,
    initial_value: ContextualInitialValue[_ItemT],
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def scan(
    group: BlockGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: ScanOperator | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: ContextualInitialValue[_ScalarT] | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT: ...
@overload
def scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: NonSumScanOperator,
    initial_value: ContextualInitialValue[_ScalarT],
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT: ...
@overload
def scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: ScanOperator | None = None,
    initial_value: None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT: ...
@overload
def exclusive_sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT: ...
@overload
def exclusive_sum(
    group: BlockGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def exclusive_sum(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT: ...
@overload
def inclusive_sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT: ...
@overload
def inclusive_sum(
    group: BlockGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def inclusive_sum(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: ContextualInitialValue[_ScalarT] | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: NonSumScanOperator,
    initial_value: ContextualInitialValue[_ScalarT],
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: ContextualInitialValue[_ItemT] | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    scan_op: NonSumScanOperator,
    initial_value: ContextualInitialValue[_ItemT],
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: ContextualInitialValue[_ScalarT] | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: NonSumScanOperator,
    initial_value: ContextualInitialValue[_ScalarT],
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT: ...
@overload
def inclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: ScanOperator | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT: ...
@overload
def inclusive_scan(
    group: BlockGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    scan_op: ScanOperator | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def inclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: ScanOperator | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT: ...
