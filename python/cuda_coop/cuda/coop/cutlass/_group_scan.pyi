# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS scans."""

from __future__ import annotations

from typing import Any, Literal, overload

from .._typing import ScanAlgorithm as _ScanAlgorithm
from .._typing import ScanOperator as _ScanOperator
from .._typing import _NonSumScanOperator as _NonSumScanOperator
from .._typing import _PortableNumericScalar as _PortableNumericScalar
from .._typing import _SumScanOperator as _SumScanOperator
from .._typing import _ValidItems as _ValidItems
from ._types import TempStorage, ThreadData
from ._types import _BlockGroup as _BlockGroup
from ._types import _CutlassNumericT as _CutlassNumericT
from ._types import _CutlassTensorSample as _CutlassTensorSample
from ._types import _CutlassTensorSSASample as _CutlassTensorSSASample
from ._types import _ScalarT as _ScalarT
from ._types import _WarpGroup as _WarpGroup

@overload
def scan(
    group: _BlockGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[_CutlassNumericT]:
    """Return block-exclusive sums without mutating the input payload."""

@overload
def scan(
    group: _BlockGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[_CutlassNumericT]:
    """Return non-sum block prefixes from a required initial value."""

@overload
def scan(
    group: _BlockGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[_CutlassNumericT]:
    """Return block-inclusive prefixes without mutating the input payload."""

@overload
def scan(
    group: _BlockGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[Any]:
    """Return block-exclusive sums for a CUTLASS register tensor.

    ``group`` is a complete block and ``value`` is an rmem ``Tensor`` or
    ``TensorSSA`` with a static item count. ``mode`` selects exclusive output;
    ``scan_op`` selects sum and ``initial_value`` optionally supplies its
    initial value. ``algorithm`` selects the block implementation and
    ``temp_storage`` supplies scratch. ``valid_items`` must remain ``None``;
    ``aggregate_output`` receives the input aggregate. The compiler validates
    the element dtype; static analysis represents the
    returned ``ThreadData`` element type as ``Any``.
    """

@overload
def scan(
    group: _BlockGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[Any]:
    """Return non-sum block prefixes for a CUTLASS register tensor.

    ``group`` is a complete block and ``value`` is an rmem ``Tensor`` or
    ``TensorSSA`` with a static item count. ``mode`` selects exclusive output;
    ``scan_op`` selects a non-sum operation and ``initial_value`` supplies its
    required initial value. ``algorithm`` selects the block implementation,
    ``temp_storage`` supplies scratch. ``valid_items`` must remain ``None``;
    ``aggregate_output`` receives the input aggregate. The compiler validates
    the element dtype; static analysis represents the returned ``ThreadData``
    element type as ``Any``.
    """

@overload
def scan(
    group: _BlockGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[Any]:
    """Return block-inclusive prefixes for a CUTLASS register tensor.

    ``group`` is a complete block and ``value`` is an rmem ``Tensor`` or
    ``TensorSSA`` with a static item count. ``mode`` selects inclusive output;
    ``scan_op`` selects the built-in operation and ``initial_value`` must remain
    absent. ``algorithm`` selects the block implementation and ``temp_storage``
    supplies scratch. ``valid_items`` must remain ``None``;
    ``aggregate_output`` receives the input aggregate. The compiler validates
    the element dtype; static analysis represents the
    returned ``ThreadData`` element type as ``Any``.
    """

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
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Block-exclusive sum a CUTLASS scalar from an optional initial value."""

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
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Block-exclusive scan a CUTLASS scalar with a non-sum initial value."""

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
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Block-inclusive scan a portable or structural CUTLASS scalar."""

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
    valid_items: _ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Warp-group-exclusive sum a scalar from an optional initial value."""

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
    valid_items: _ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Warp-group-exclusive scan with a required non-sum initial value."""

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
    valid_items: _ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Warp-group-inclusive scan a scalar without an initial value."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[_CutlassNumericT]:
    """Return block-exclusive sums with the input payload shape."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[Any]:
    """Return block-exclusive sums for a CUTLASS register tensor.

    ``group`` is a complete block and ``value`` supplies a static rmem or SSA
    register payload. ``algorithm`` selects the block implementation and
    ``temp_storage`` supplies scratch. ``valid_items`` must remain ``None``;
    ``aggregate_output`` receives the input aggregate. The compiler validates
    the element dtype; the external dtype remains ``Any`` to static analysis.
    """

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Preserve a scalar type through block-exclusive sum."""

@overload
def exclusive_sum(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp exclusive sum."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[_CutlassNumericT]:
    """Return block-inclusive sums with the input payload shape."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[Any]:
    """Return block-inclusive sums for a CUTLASS register tensor.

    ``group`` is a complete block and ``value`` supplies a static rmem or SSA
    register payload. ``algorithm`` selects the block implementation and
    ``temp_storage`` supplies scratch. ``valid_items`` must remain ``None``;
    ``aggregate_output`` receives the input aggregate. The compiler validates
    the element dtype; the external dtype remains ``Any`` to static analysis.
    """

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Preserve a scalar type through block-inclusive sum."""

@overload
def inclusive_sum(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp inclusive sum."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[_CutlassNumericT]:
    """Return block-exclusive sums with the input payload shape."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[_CutlassNumericT]:
    """Return non-sum block prefixes from a required initial value."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[Any]:
    """Return block-exclusive sums for a CUTLASS register tensor.

    ``group`` is a complete block and ``value`` supplies a static rmem or SSA
    register payload. ``scan_op`` selects sum and ``initial_value`` optionally
    supplies its initial value, ``algorithm`` selects the block implementation,
    and ``temp_storage`` supplies scratch. ``valid_items`` must remain ``None``;
    ``aggregate_output`` receives the input aggregate. The compiler validates
    the element dtype; the external dtype remains ``Any`` to static analysis.
    """

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[Any]:
    """Return non-sum block prefixes for a CUTLASS register tensor.

    ``group`` is a complete block and ``value`` supplies a static rmem or SSA
    register payload. ``scan_op`` selects the operation and ``initial_value``
    supplies its required initial value, ``algorithm`` selects the block
    implementation, and ``temp_storage`` supplies scratch. ``valid_items``
    must remain ``None``; ``aggregate_output`` receives the
    input aggregate. The compiler validates the element dtype; the
    external dtype remains ``Any`` to static analysis.
    """

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Block-exclusive sum a scalar from an optional initial value."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Block-exclusive scan a scalar with a required non-sum initial value."""

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
    valid_items: _ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Warp-group-exclusive sum a scalar from an optional initial value."""

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
    valid_items: _ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Warp-group-exclusive scan with a required non-sum initial value."""

@overload
def inclusive_scan(
    group: _BlockGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    scan_op: _ScanOperator | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[_CutlassNumericT]:
    """Return block-inclusive prefixes with the input payload shape."""

@overload
def inclusive_scan(
    group: _BlockGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    scan_op: _ScanOperator | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[Any]:
    """Return block-inclusive prefixes for a CUTLASS register tensor.

    ``group`` is a complete block and ``value`` supplies a static rmem or SSA
    register payload. ``scan_op`` selects the built-in operation, ``algorithm``
    selects the block implementation and ``temp_storage`` supplies scratch.
    ``valid_items`` must remain ``None``; ``aggregate_output`` receives the
    input aggregate. The compiler validates the
    element dtype; the external dtype remains ``Any`` to static analysis.
    """

@overload
def inclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _ScanOperator | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Preserve a scalar type through block-inclusive Scan."""

@overload
def inclusive_scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _ScanOperator | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp inclusive Scan."""

__all__ = [
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "scan",
]
