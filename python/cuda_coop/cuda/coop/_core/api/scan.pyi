# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable scan family."""

from typing import Literal, overload

from typing_extensions import TypeVar

from cuda.coop._typing import ScanAlgorithm as _ScanAlgorithm
from cuda.coop._typing import ScanOperator as _ScanOperator
from cuda.coop._typing import TempStorageLike as TempStorageLike
from cuda.coop._typing import ThreadDataLike as ThreadDataLike
from cuda.coop._typing import _NonSumScanOperator as _NonSumScanOperator
from cuda.coop._typing import _PortableNumericScalar as _PortableNumericScalar
from cuda.coop._typing import _SumScanOperator as _SumScanOperator

from .thread_group import _BlockGroup, _WarpGroup

_PortableNumericT = TypeVar("_PortableNumericT", bound=_PortableNumericScalar)
_ScalarT = TypeVar("_ScalarT", bound=_PortableNumericScalar)

@overload
def scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-exclusive sum a scalar, optionally from an initial value.

    External compiler scalar values typed as ``Any`` necessarily return
    ``Any`` in the backend-neutral static contract.
    """

@overload
def scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-exclusive scan a scalar with a required non-sum initial value."""

@overload
def scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-inclusive scan a portable or structural compiler scalar."""

@overload
def scan(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-exclusive sums without mutating the input payload."""

@overload
def scan(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return non-sum block prefixes from a required initial value."""

@overload
def scan(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-inclusive prefixes without mutating the input payload."""

@overload
def scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-exclusive sum a scalar from an optional initial value.

    Physical- and logical-warp scans have no algorithm selector or caller-owned scratch.
    """

@overload
def scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-exclusive scan with a non-sum initial value."""

@overload
def scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | None = None,
    initial_value: None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-inclusive scan without an initial value."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Preserve a scalar type through block-exclusive sum."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-exclusive sums with the input payload shape."""

@overload
def exclusive_sum(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp exclusive sum."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Preserve a scalar type through block-inclusive sum."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-inclusive sums with the input payload shape."""

@overload
def inclusive_sum(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp inclusive sum."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-exclusive sum a scalar, optionally from an initial value."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-exclusive scan a scalar with a required non-sum initial value."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-exclusive sums with the input payload shape."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return non-sum block prefixes from a required initial value."""

@overload
def exclusive_scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-exclusive sum from an optional initial value."""

@overload
def exclusive_scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-exclusive scan with a non-sum initial value."""

@overload
def inclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _ScanOperator | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Preserve a scalar type through block-inclusive Scan."""

@overload
def inclusive_scan(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    scan_op: _ScanOperator | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-inclusive prefixes with the input payload shape."""

@overload
def inclusive_scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _ScanOperator | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp inclusive Scan."""
