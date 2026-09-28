# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Literal, overload

from typing_extensions import TypedDict, TypeVar, Unpack

from cuda.coop._scan_typing import (
    _CompilerFloat32T,
    _CompilerFloat64T,
    _CompilerInt8T,
    _CompilerInt16T,
    _CompilerInt32T,
    _CompilerInt64T,
    _CompilerUint8T,
    _CompilerUint16T,
    _CompilerUint32T,
    _CompilerUint64T,
    _CutlassFloat32,
    _CutlassFloat64,
    _CutlassInt8,
    _CutlassInt16,
    _CutlassInt32,
    _CutlassInt64,
    _CutlassUint8,
    _CutlassUint16,
    _CutlassUint32,
    _CutlassUint64,
    _NumpyFloat32T,
    _NumpyFloat64T,
    _NumpyInt8T,
    _NumpyInt16T,
    _NumpyInt32T,
    _NumpyInt64T,
    _NumpyUint8T,
    _NumpyUint16T,
    _NumpyUint32T,
    _NumpyUint64T,
    _SeedFloat32,
    _SeedFloat64,
    _SeedInt8,
    _SeedInt16,
    _SeedInt32,
    _SeedInt64,
    _SeedUint8,
    _SeedUint16,
    _SeedUint32,
    _SeedUint64,
)
from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ContextualInitialValue,
    NonSumScanOperator,
    ScanAlgorithm,
    ScanOperator,
    SumScanOperator,
    TempStorageLike,
    ThreadDataLike,
)

from .thread_group import BlockGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)
_ScalarT = TypeVar("_ScalarT", bound=CommonNumericScalar)

class _BlockSeededScanOptions(TypedDict, total=False):
    scan_op: ScanOperator | None
    algorithm: ScanAlgorithm | None
    temp_storage: TempStorageLike | None

class _BlockSeededScanModeOptions(_BlockSeededScanOptions, total=False):
    mode: Literal["exclusive"]

class _WarpSeededScanOptions(TypedDict, total=False):
    scan_op: ScanOperator | None
    algorithm: None
    temp_storage: None

class _WarpSeededScanModeOptions(_WarpSeededScanOptions, total=False):
    mode: Literal["exclusive"]

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
    value: _NumpyInt8T,
    /,
    *,
    initial_value: _CutlassInt8,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _NumpyInt8T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerInt8T,
    /,
    *,
    initial_value: _SeedInt8,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _CompilerInt8T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyUint8T,
    /,
    *,
    initial_value: _CutlassUint8,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _NumpyUint8T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerUint8T,
    /,
    *,
    initial_value: _SeedUint8,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _CompilerUint8T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyInt16T,
    /,
    *,
    initial_value: _CutlassInt16,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _NumpyInt16T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerInt16T,
    /,
    *,
    initial_value: _SeedInt16,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _CompilerInt16T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyUint16T,
    /,
    *,
    initial_value: _CutlassUint16,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _NumpyUint16T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerUint16T,
    /,
    *,
    initial_value: _SeedUint16,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _CompilerUint16T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyInt32T,
    /,
    *,
    initial_value: _CutlassInt32,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _NumpyInt32T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerInt32T,
    /,
    *,
    initial_value: _SeedInt32,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _CompilerInt32T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyUint32T,
    /,
    *,
    initial_value: _CutlassUint32,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _NumpyUint32T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerUint32T,
    /,
    *,
    initial_value: _SeedUint32,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _CompilerUint32T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyInt64T,
    /,
    *,
    initial_value: _CutlassInt64,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _NumpyInt64T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerInt64T,
    /,
    *,
    initial_value: _SeedInt64,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _CompilerInt64T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyUint64T,
    /,
    *,
    initial_value: _CutlassUint64,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _NumpyUint64T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerUint64T,
    /,
    *,
    initial_value: _SeedUint64,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _CompilerUint64T: ...
@overload
def scan(  # type: ignore[overload-overlap, unused-ignore]
    group: BlockGroup,
    value: _NumpyFloat32T,
    /,
    *,
    initial_value: _CutlassFloat32,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _NumpyFloat32T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerFloat32T,
    /,
    *,
    initial_value: _SeedFloat32,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _CompilerFloat32T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyFloat64T,
    /,
    *,
    initial_value: _CutlassFloat64,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _NumpyFloat64T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerFloat64T,
    /,
    *,
    initial_value: _SeedFloat64,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> _CompilerFloat64T: ...
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
    value: CommonThreadDataLike[_ItemT],
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
    value: CommonThreadDataLike[_NumpyInt8T],
    /,
    *,
    initial_value: _CutlassInt8,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_NumpyInt8T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt8T],
    /,
    *,
    initial_value: _SeedInt8,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_CompilerInt8T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint8T],
    /,
    *,
    initial_value: _CutlassUint8,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_NumpyUint8T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint8T],
    /,
    *,
    initial_value: _SeedUint8,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_CompilerUint8T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt16T],
    /,
    *,
    initial_value: _CutlassInt16,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_NumpyInt16T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt16T],
    /,
    *,
    initial_value: _SeedInt16,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_CompilerInt16T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint16T],
    /,
    *,
    initial_value: _CutlassUint16,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_NumpyUint16T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint16T],
    /,
    *,
    initial_value: _SeedUint16,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_CompilerUint16T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt32T],
    /,
    *,
    initial_value: _CutlassInt32,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_NumpyInt32T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt32T],
    /,
    *,
    initial_value: _SeedInt32,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_CompilerInt32T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint32T],
    /,
    *,
    initial_value: _CutlassUint32,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_NumpyUint32T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint32T],
    /,
    *,
    initial_value: _SeedUint32,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_CompilerUint32T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt64T],
    /,
    *,
    initial_value: _CutlassInt64,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_NumpyInt64T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt64T],
    /,
    *,
    initial_value: _SeedInt64,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_CompilerInt64T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint64T],
    /,
    *,
    initial_value: _CutlassUint64,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_NumpyUint64T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint64T],
    /,
    *,
    initial_value: _SeedUint64,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_CompilerUint64T]: ...

# Without CUTLASS, its guarded float protocols coincide. The input dtype
# still preserves this return type; callers cannot pass NumPy seeds here.
@overload
def scan(  # type: ignore[overload-overlap, unused-ignore]
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyFloat32T],
    /,
    *,
    initial_value: _CutlassFloat32,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_NumpyFloat32T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerFloat32T],
    /,
    *,
    initial_value: _SeedFloat32,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_CompilerFloat32T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyFloat64T],
    /,
    *,
    initial_value: _CutlassFloat64,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_NumpyFloat64T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerFloat64T],
    /,
    *,
    initial_value: _SeedFloat64,
    **kwargs: Unpack[_BlockSeededScanModeOptions],
) -> ThreadDataLike[_CompilerFloat64T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
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
    value: CommonThreadDataLike[_ItemT],
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
    value: _NumpyInt8T,
    /,
    *,
    initial_value: _CutlassInt8,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _NumpyInt8T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerInt8T,
    /,
    *,
    initial_value: _SeedInt8,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _CompilerInt8T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyUint8T,
    /,
    *,
    initial_value: _CutlassUint8,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _NumpyUint8T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerUint8T,
    /,
    *,
    initial_value: _SeedUint8,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _CompilerUint8T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyInt16T,
    /,
    *,
    initial_value: _CutlassInt16,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _NumpyInt16T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerInt16T,
    /,
    *,
    initial_value: _SeedInt16,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _CompilerInt16T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyUint16T,
    /,
    *,
    initial_value: _CutlassUint16,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _NumpyUint16T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerUint16T,
    /,
    *,
    initial_value: _SeedUint16,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _CompilerUint16T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyInt32T,
    /,
    *,
    initial_value: _CutlassInt32,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _NumpyInt32T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerInt32T,
    /,
    *,
    initial_value: _SeedInt32,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _CompilerInt32T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyUint32T,
    /,
    *,
    initial_value: _CutlassUint32,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _NumpyUint32T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerUint32T,
    /,
    *,
    initial_value: _SeedUint32,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _CompilerUint32T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyInt64T,
    /,
    *,
    initial_value: _CutlassInt64,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _NumpyInt64T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerInt64T,
    /,
    *,
    initial_value: _SeedInt64,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _CompilerInt64T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyUint64T,
    /,
    *,
    initial_value: _CutlassUint64,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _NumpyUint64T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerUint64T,
    /,
    *,
    initial_value: _SeedUint64,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _CompilerUint64T: ...
@overload
def scan(  # type: ignore[overload-overlap, unused-ignore]
    group: WarpGroup,
    value: _NumpyFloat32T,
    /,
    *,
    initial_value: _CutlassFloat32,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _NumpyFloat32T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerFloat32T,
    /,
    *,
    initial_value: _SeedFloat32,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _CompilerFloat32T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyFloat64T,
    /,
    *,
    initial_value: _CutlassFloat64,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _NumpyFloat64T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerFloat64T,
    /,
    *,
    initial_value: _SeedFloat64,
    **kwargs: Unpack[_WarpSeededScanModeOptions],
) -> _CompilerFloat64T: ...
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
    value: CommonThreadDataLike[_ItemT],
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
    value: CommonThreadDataLike[_ItemT],
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
    value: _NumpyInt8T,
    /,
    *,
    initial_value: _CutlassInt8,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _NumpyInt8T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerInt8T,
    /,
    *,
    initial_value: _SeedInt8,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _CompilerInt8T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyUint8T,
    /,
    *,
    initial_value: _CutlassUint8,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _NumpyUint8T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerUint8T,
    /,
    *,
    initial_value: _SeedUint8,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _CompilerUint8T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyInt16T,
    /,
    *,
    initial_value: _CutlassInt16,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _NumpyInt16T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerInt16T,
    /,
    *,
    initial_value: _SeedInt16,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _CompilerInt16T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyUint16T,
    /,
    *,
    initial_value: _CutlassUint16,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _NumpyUint16T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerUint16T,
    /,
    *,
    initial_value: _SeedUint16,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _CompilerUint16T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyInt32T,
    /,
    *,
    initial_value: _CutlassInt32,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _NumpyInt32T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerInt32T,
    /,
    *,
    initial_value: _SeedInt32,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _CompilerInt32T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyUint32T,
    /,
    *,
    initial_value: _CutlassUint32,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _NumpyUint32T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerUint32T,
    /,
    *,
    initial_value: _SeedUint32,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _CompilerUint32T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyInt64T,
    /,
    *,
    initial_value: _CutlassInt64,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _NumpyInt64T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerInt64T,
    /,
    *,
    initial_value: _SeedInt64,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _CompilerInt64T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyUint64T,
    /,
    *,
    initial_value: _CutlassUint64,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _NumpyUint64T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerUint64T,
    /,
    *,
    initial_value: _SeedUint64,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _CompilerUint64T: ...
@overload
def exclusive_scan(  # type: ignore[overload-overlap, unused-ignore]
    group: BlockGroup,
    value: _NumpyFloat32T,
    /,
    *,
    initial_value: _CutlassFloat32,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _NumpyFloat32T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerFloat32T,
    /,
    *,
    initial_value: _SeedFloat32,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _CompilerFloat32T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyFloat64T,
    /,
    *,
    initial_value: _CutlassFloat64,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _NumpyFloat64T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerFloat64T,
    /,
    *,
    initial_value: _SeedFloat64,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> _CompilerFloat64T: ...
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
    value: CommonThreadDataLike[_ItemT],
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
    value: CommonThreadDataLike[_NumpyInt8T],
    /,
    *,
    initial_value: _CutlassInt8,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_NumpyInt8T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt8T],
    /,
    *,
    initial_value: _SeedInt8,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_CompilerInt8T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint8T],
    /,
    *,
    initial_value: _CutlassUint8,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_NumpyUint8T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint8T],
    /,
    *,
    initial_value: _SeedUint8,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_CompilerUint8T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt16T],
    /,
    *,
    initial_value: _CutlassInt16,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_NumpyInt16T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt16T],
    /,
    *,
    initial_value: _SeedInt16,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_CompilerInt16T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint16T],
    /,
    *,
    initial_value: _CutlassUint16,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_NumpyUint16T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint16T],
    /,
    *,
    initial_value: _SeedUint16,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_CompilerUint16T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt32T],
    /,
    *,
    initial_value: _CutlassInt32,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_NumpyInt32T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt32T],
    /,
    *,
    initial_value: _SeedInt32,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_CompilerInt32T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint32T],
    /,
    *,
    initial_value: _CutlassUint32,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_NumpyUint32T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint32T],
    /,
    *,
    initial_value: _SeedUint32,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_CompilerUint32T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt64T],
    /,
    *,
    initial_value: _CutlassInt64,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_NumpyInt64T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt64T],
    /,
    *,
    initial_value: _SeedInt64,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_CompilerInt64T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint64T],
    /,
    *,
    initial_value: _CutlassUint64,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_NumpyUint64T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint64T],
    /,
    *,
    initial_value: _SeedUint64,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_CompilerUint64T]: ...

# See the corresponding scan overload for the optional-CUTLASS overlap.
@overload
def exclusive_scan(  # type: ignore[overload-overlap, unused-ignore]
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyFloat32T],
    /,
    *,
    initial_value: _CutlassFloat32,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_NumpyFloat32T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerFloat32T],
    /,
    *,
    initial_value: _SeedFloat32,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_CompilerFloat32T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyFloat64T],
    /,
    *,
    initial_value: _CutlassFloat64,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_NumpyFloat64T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerFloat64T],
    /,
    *,
    initial_value: _SeedFloat64,
    **kwargs: Unpack[_BlockSeededScanOptions],
) -> ThreadDataLike[_CompilerFloat64T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
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
    value: _NumpyInt8T,
    /,
    *,
    initial_value: _CutlassInt8,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _NumpyInt8T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerInt8T,
    /,
    *,
    initial_value: _SeedInt8,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _CompilerInt8T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyUint8T,
    /,
    *,
    initial_value: _CutlassUint8,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _NumpyUint8T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerUint8T,
    /,
    *,
    initial_value: _SeedUint8,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _CompilerUint8T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyInt16T,
    /,
    *,
    initial_value: _CutlassInt16,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _NumpyInt16T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerInt16T,
    /,
    *,
    initial_value: _SeedInt16,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _CompilerInt16T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyUint16T,
    /,
    *,
    initial_value: _CutlassUint16,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _NumpyUint16T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerUint16T,
    /,
    *,
    initial_value: _SeedUint16,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _CompilerUint16T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyInt32T,
    /,
    *,
    initial_value: _CutlassInt32,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _NumpyInt32T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerInt32T,
    /,
    *,
    initial_value: _SeedInt32,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _CompilerInt32T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyUint32T,
    /,
    *,
    initial_value: _CutlassUint32,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _NumpyUint32T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerUint32T,
    /,
    *,
    initial_value: _SeedUint32,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _CompilerUint32T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyInt64T,
    /,
    *,
    initial_value: _CutlassInt64,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _NumpyInt64T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerInt64T,
    /,
    *,
    initial_value: _SeedInt64,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _CompilerInt64T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyUint64T,
    /,
    *,
    initial_value: _CutlassUint64,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _NumpyUint64T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerUint64T,
    /,
    *,
    initial_value: _SeedUint64,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _CompilerUint64T: ...
@overload
def exclusive_scan(  # type: ignore[overload-overlap, unused-ignore]
    group: WarpGroup,
    value: _NumpyFloat32T,
    /,
    *,
    initial_value: _CutlassFloat32,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _NumpyFloat32T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerFloat32T,
    /,
    *,
    initial_value: _SeedFloat32,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _CompilerFloat32T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyFloat64T,
    /,
    *,
    initial_value: _CutlassFloat64,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _NumpyFloat64T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerFloat64T,
    /,
    *,
    initial_value: _SeedFloat64,
    **kwargs: Unpack[_WarpSeededScanOptions],
) -> _CompilerFloat64T: ...
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
    value: CommonThreadDataLike[_ItemT],
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
