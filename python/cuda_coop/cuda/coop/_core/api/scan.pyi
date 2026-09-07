# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable scan family."""

from typing import Literal, overload

from typing_extensions import TypeVar

from cuda.coop._typing import (
    NonSumScanOperator,
    PortableNumericScalar,
    ScanAlgorithm,
    ScanOperator,
    SumScanOperator,
    TempStorageLike,
    ThreadDataLike,
)

from .thread_group import BlockGroup, WarpGroup

_PortableNumericT = TypeVar("_PortableNumericT", bound=PortableNumericScalar)
_ScalarT = TypeVar("_ScalarT", bound=PortableNumericScalar)

@overload
def scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-exclusive sum a scalar, optionally from an initial value.

    External compiler scalar values typed as ``Any`` necessarily return
    ``Any`` in the backend-neutral static contract.
    """

@overload
def scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-exclusive scan a scalar with a required non-sum initial value."""

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
) -> _ScalarT:
    """Block-inclusive scan a portable or structural compiler scalar."""

@overload
def scan(
    group: BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-exclusive sums without mutating the input payload."""

@overload
def scan(
    group: BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return non-sum block prefixes from a required initial value."""

@overload
def scan(
    group: BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: ScanOperator | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-inclusive prefixes without mutating the input payload."""

@overload
def scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-exclusive sum a scalar from an optional initial value.

    Physical- and logical-warp scans have no algorithm selector or caller-owned scratch.
    """

@overload
def scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-exclusive scan with a non-sum initial value."""

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
) -> _ScalarT:
    """Physical- or logical-warp-inclusive scan without an initial value."""

@overload
def exclusive_sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Preserve a scalar type through block-exclusive sum."""

@overload
def exclusive_sum(
    group: BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-exclusive sums with the input payload shape."""

@overload
def exclusive_sum(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp exclusive sum."""

@overload
def inclusive_sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Preserve a scalar type through block-inclusive sum."""

@overload
def inclusive_sum(
    group: BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-inclusive sums with the input payload shape."""

@overload
def inclusive_sum(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp inclusive sum."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-exclusive sum a scalar, optionally from an initial value."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-exclusive scan a scalar with a required non-sum initial value."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-exclusive sums with the input payload shape."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return non-sum block prefixes from a required initial value."""

@overload
def exclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-exclusive sum from an optional initial value."""

@overload
def exclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-exclusive scan with a non-sum initial value."""

@overload
def inclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: ScanOperator | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Preserve a scalar type through block-inclusive Scan."""

@overload
def inclusive_scan(
    group: BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    scan_op: ScanOperator | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-inclusive prefixes with the input payload shape."""

@overload
def inclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: ScanOperator | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp inclusive Scan."""
