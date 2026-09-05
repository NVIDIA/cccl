# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS scans."""

from __future__ import annotations

from typing import Any, Literal, overload

from .._typing import (
    NonSumScanOperator,
    PortableNumericScalar,
    ScanAlgorithm,
    ScanOperator,
    SumScanOperator,
    ValidItems,
)
from ._temp_storage import TempStorage
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData
from ._thread_group import BlockGroup, WarpGroup
from ._typing import CutlassNumericT, ScalarT

@overload
def scan(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[CutlassNumericT]:
    """Return block-exclusive sums without mutating the input payload."""

@overload
def scan(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[CutlassNumericT]:
    """Return non-sum block prefixes from a required initial value."""

@overload
def scan(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: ScanOperator | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[CutlassNumericT]:
    """Return block-inclusive prefixes without mutating the input payload."""

@overload
def scan(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
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
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: ScanAlgorithm | None = None,
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
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: ScanOperator | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | None = None,
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
    group: BlockGroup,
    value: ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Block-exclusive sum a CUTLASS scalar from an optional initial value."""

@overload
def scan(
    group: BlockGroup,
    value: ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Block-exclusive scan a CUTLASS scalar with a non-sum initial value."""

@overload
def scan(
    group: BlockGroup,
    value: ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: ScanOperator | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Block-inclusive scan a portable or structural CUTLASS scalar."""

@overload
def scan(
    group: WarpGroup,
    value: ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Warp-group-exclusive sum a scalar from an optional initial value."""

@overload
def scan(
    group: WarpGroup,
    value: ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Warp-group-exclusive scan with a required non-sum initial value."""

@overload
def scan(
    group: WarpGroup,
    value: ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: ScanOperator | None = None,
    initial_value: None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Warp-group-inclusive scan a scalar without an initial value."""

@overload
def exclusive_sum(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[CutlassNumericT]:
    """Return block-exclusive sums with the input payload shape."""

@overload
def exclusive_sum(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[Any]:
    """Return block-exclusive sums for a CUTLASS register tensor.

    ``group`` is a complete block and ``value`` supplies a static rmem or SSA
    register payload. ``algorithm`` selects the block implementation and
    ``temp_storage`` supplies scratch. ``aggregate_output`` receives the input
    aggregate. The compiler validates the element dtype; the external dtype
    remains ``Any`` to static analysis.
    """

@overload
def exclusive_sum(
    group: BlockGroup,
    value: ScalarT,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Preserve a scalar type through block-exclusive sum."""

@overload
def exclusive_sum(
    group: WarpGroup,
    value: ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Preserve a scalar type through physical- or logical-warp exclusive sum."""

@overload
def inclusive_sum(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[CutlassNumericT]:
    """Return block-inclusive sums with the input payload shape."""

@overload
def inclusive_sum(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
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
    group: BlockGroup,
    value: ScalarT,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Preserve a scalar type through block-inclusive sum."""

@overload
def inclusive_sum(
    group: WarpGroup,
    value: ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Preserve a scalar type through physical- or logical-warp inclusive sum."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[CutlassNumericT]:
    """Return block-exclusive sums with the input payload shape."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[CutlassNumericT]:
    """Return non-sum block prefixes from a required initial value."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
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
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: ScanAlgorithm | None = None,
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
    group: BlockGroup,
    value: ScalarT,
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Block-exclusive sum a scalar from an optional initial value."""

@overload
def exclusive_scan(
    group: BlockGroup,
    value: ScalarT,
    /,
    *,
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Block-exclusive scan a scalar with a required non-sum initial value."""

@overload
def exclusive_scan(
    group: WarpGroup,
    value: ScalarT,
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: PortableNumericScalar | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Warp-group-exclusive sum a scalar from an optional initial value."""

@overload
def exclusive_scan(
    group: WarpGroup,
    value: ScalarT,
    /,
    *,
    scan_op: NonSumScanOperator,
    initial_value: PortableNumericScalar,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Warp-group-exclusive scan with a required non-sum initial value."""

@overload
def inclusive_scan(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    scan_op: ScanOperator | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData[CutlassNumericT]:
    """Return block-inclusive prefixes with the input payload shape."""

@overload
def inclusive_scan(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    scan_op: ScanOperator | None = None,
    algorithm: ScanAlgorithm | None = None,
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
    group: BlockGroup,
    value: ScalarT,
    /,
    *,
    scan_op: ScanOperator | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Preserve a scalar type through block-inclusive Scan."""

@overload
def inclusive_scan(
    group: WarpGroup,
    value: ScalarT,
    /,
    *,
    scan_op: ScanOperator | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> ScalarT:
    """Preserve a scalar type through physical- or logical-warp inclusive Scan."""

__all__ = [
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "scan",
]
