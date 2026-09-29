# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Built-in Scan signatures for the qualified CUTLASS backend."""

from collections.abc import Callable
from typing import Any, Generic, Literal, Protocol, TypeAlias, overload

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
    ValidItems,
)

from .._core.api.thread_group import BlockGroup, WarpGroup
from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
)

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)
_ScalarT = TypeVar("_ScalarT", bound=CommonNumericScalar)
_RegisterPayload: TypeAlias = CutlassTensorSample | CutlassTensorSSASample
_NumpyScanUfuncName: TypeAlias = Literal[
    "add",
    "multiply",
    "minimum",
    "maximum",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
]

class _NumpyScanUfunc(Protocol):
    @property
    def __name__(self) -> _NumpyScanUfuncName: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[1]: ...

class _NumpySumScanUfunc(_NumpyScanUfunc, Protocol):
    @property
    def __name__(self) -> Literal["add"]: ...

# The compiler accepts known identities only, not arbitrary callbacks.
_OperatorScanAlias: TypeAlias = Callable[[object, object], object]
_BuiltinScanOperator: TypeAlias = (
    ScanOperator | _OperatorScanAlias | _NumpyScanUfunc
)
_SeededScanOperator: TypeAlias = (
    NonSumScanOperator | _OperatorScanAlias | _NumpyScanUfunc
)

class _BlockSeededScanOptions(TypedDict, Generic[_ItemT], total=False):
    scan_op: _BuiltinScanOperator | None
    algorithm: ScanAlgorithm | None
    temp_storage: TempStorageLike | None
    valid_items: None
    aggregate_output: ThreadDataLike[_ItemT] | None

class _BlockSeededScanModeOptions(_BlockSeededScanOptions[_ItemT], total=False):
    mode: Literal["exclusive"]

class _WarpSeededScanOptions(TypedDict, Generic[_ItemT], total=False):
    scan_op: _BuiltinScanOperator | None
    algorithm: None
    temp_storage: None
    valid_items: ValidItems | None
    aggregate_output: ThreadDataLike[_ItemT] | None

class _WarpSeededScanModeOptions(_WarpSeededScanOptions[_ItemT], total=False):
    mode: Literal["exclusive"]

@overload
def scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | _NumpySumScanUfunc | None = None,
    initial_value: ContextualInitialValue[_ScalarT] | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyInt8T,
    /,
    *,
    initial_value: _CutlassInt8,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyInt8T]],
) -> _NumpyInt8T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerInt8T,
    /,
    *,
    initial_value: _SeedInt8,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerInt8T]],
) -> _CompilerInt8T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyUint8T,
    /,
    *,
    initial_value: _CutlassUint8,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyUint8T]],
) -> _NumpyUint8T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerUint8T,
    /,
    *,
    initial_value: _SeedUint8,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerUint8T]],
) -> _CompilerUint8T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyInt16T,
    /,
    *,
    initial_value: _CutlassInt16,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyInt16T]],
) -> _NumpyInt16T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerInt16T,
    /,
    *,
    initial_value: _SeedInt16,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerInt16T]],
) -> _CompilerInt16T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyUint16T,
    /,
    *,
    initial_value: _CutlassUint16,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyUint16T]],
) -> _NumpyUint16T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerUint16T,
    /,
    *,
    initial_value: _SeedUint16,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerUint16T]],
) -> _CompilerUint16T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyInt32T,
    /,
    *,
    initial_value: _CutlassInt32,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyInt32T]],
) -> _NumpyInt32T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerInt32T,
    /,
    *,
    initial_value: _SeedInt32,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerInt32T]],
) -> _CompilerInt32T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyUint32T,
    /,
    *,
    initial_value: _CutlassUint32,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyUint32T]],
) -> _NumpyUint32T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerUint32T,
    /,
    *,
    initial_value: _SeedUint32,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerUint32T]],
) -> _CompilerUint32T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyInt64T,
    /,
    *,
    initial_value: _CutlassInt64,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyInt64T]],
) -> _NumpyInt64T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerInt64T,
    /,
    *,
    initial_value: _SeedInt64,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerInt64T]],
) -> _CompilerInt64T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyUint64T,
    /,
    *,
    initial_value: _CutlassUint64,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyUint64T]],
) -> _NumpyUint64T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerUint64T,
    /,
    *,
    initial_value: _SeedUint64,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerUint64T]],
) -> _CompilerUint64T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyFloat32T,
    /,
    *,
    initial_value: _CutlassFloat32,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyFloat32T]],
) -> _NumpyFloat32T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerFloat32T,
    /,
    *,
    initial_value: _SeedFloat32,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerFloat32T]],
) -> _CompilerFloat32T: ...
@overload
def scan(
    group: BlockGroup,
    value: _NumpyFloat64T,
    /,
    *,
    initial_value: _CutlassFloat64,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyFloat64T]],
) -> _NumpyFloat64T: ...
@overload
def scan(
    group: BlockGroup,
    value: _CompilerFloat64T,
    /,
    *,
    initial_value: _SeedFloat64,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerFloat64T]],
) -> _CompilerFloat64T: ...
@overload
def scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SeededScanOperator,
    initial_value: ContextualInitialValue[_ScalarT],
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _BuiltinScanOperator | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | _NumpySumScanUfunc | None = None,
    initial_value: ContextualInitialValue[_ItemT] | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadData[_ItemT]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt8T],
    /,
    *,
    initial_value: _CutlassInt8,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyInt8T]],
) -> ThreadData[_NumpyInt8T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt8T],
    /,
    *,
    initial_value: _SeedInt8,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerInt8T]],
) -> ThreadData[_CompilerInt8T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint8T],
    /,
    *,
    initial_value: _CutlassUint8,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyUint8T]],
) -> ThreadData[_NumpyUint8T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint8T],
    /,
    *,
    initial_value: _SeedUint8,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerUint8T]],
) -> ThreadData[_CompilerUint8T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt16T],
    /,
    *,
    initial_value: _CutlassInt16,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyInt16T]],
) -> ThreadData[_NumpyInt16T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt16T],
    /,
    *,
    initial_value: _SeedInt16,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerInt16T]],
) -> ThreadData[_CompilerInt16T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint16T],
    /,
    *,
    initial_value: _CutlassUint16,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyUint16T]],
) -> ThreadData[_NumpyUint16T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint16T],
    /,
    *,
    initial_value: _SeedUint16,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerUint16T]],
) -> ThreadData[_CompilerUint16T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt32T],
    /,
    *,
    initial_value: _CutlassInt32,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyInt32T]],
) -> ThreadData[_NumpyInt32T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt32T],
    /,
    *,
    initial_value: _SeedInt32,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerInt32T]],
) -> ThreadData[_CompilerInt32T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint32T],
    /,
    *,
    initial_value: _CutlassUint32,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyUint32T]],
) -> ThreadData[_NumpyUint32T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint32T],
    /,
    *,
    initial_value: _SeedUint32,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerUint32T]],
) -> ThreadData[_CompilerUint32T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt64T],
    /,
    *,
    initial_value: _CutlassInt64,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyInt64T]],
) -> ThreadData[_NumpyInt64T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt64T],
    /,
    *,
    initial_value: _SeedInt64,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerInt64T]],
) -> ThreadData[_CompilerInt64T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint64T],
    /,
    *,
    initial_value: _CutlassUint64,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyUint64T]],
) -> ThreadData[_NumpyUint64T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint64T],
    /,
    *,
    initial_value: _SeedUint64,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerUint64T]],
) -> ThreadData[_CompilerUint64T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyFloat32T],
    /,
    *,
    initial_value: _CutlassFloat32,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyFloat32T]],
) -> ThreadData[_NumpyFloat32T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerFloat32T],
    /,
    *,
    initial_value: _SeedFloat32,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerFloat32T]],
) -> ThreadData[_CompilerFloat32T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyFloat64T],
    /,
    *,
    initial_value: _CutlassFloat64,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_NumpyFloat64T]],
) -> ThreadData[_NumpyFloat64T]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerFloat64T],
    /,
    *,
    initial_value: _SeedFloat64,
    **kwargs: Unpack[_BlockSeededScanModeOptions[_CompilerFloat64T]],
) -> ThreadData[_CompilerFloat64T]: ...
@overload
def scan(
    group: BlockGroup,
    value: _RegisterPayload,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | _NumpySumScanUfunc | None = None,
    initial_value: CommonNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[Any] | None = None,
) -> ThreadData[Any]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SeededScanOperator,
    initial_value: ContextualInitialValue[_ItemT],
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadData[_ItemT]: ...
@overload
def scan(
    group: BlockGroup,
    value: _RegisterPayload,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SeededScanOperator,
    initial_value: CommonNumericScalar,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[Any] | None = None,
) -> ThreadData[Any]: ...
@overload
def scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _BuiltinScanOperator | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadData[_ItemT]: ...
@overload
def scan(
    group: BlockGroup,
    value: _RegisterPayload,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _BuiltinScanOperator | None = None,
    initial_value: None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[Any] | None = None,
) -> ThreadData[Any]: ...
@overload
def scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | _NumpySumScanUfunc | None = None,
    initial_value: ContextualInitialValue[_ScalarT] | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyInt8T,
    /,
    *,
    initial_value: _CutlassInt8,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_NumpyInt8T]],
) -> _NumpyInt8T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerInt8T,
    /,
    *,
    initial_value: _SeedInt8,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_CompilerInt8T]],
) -> _CompilerInt8T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyUint8T,
    /,
    *,
    initial_value: _CutlassUint8,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_NumpyUint8T]],
) -> _NumpyUint8T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerUint8T,
    /,
    *,
    initial_value: _SeedUint8,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_CompilerUint8T]],
) -> _CompilerUint8T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyInt16T,
    /,
    *,
    initial_value: _CutlassInt16,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_NumpyInt16T]],
) -> _NumpyInt16T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerInt16T,
    /,
    *,
    initial_value: _SeedInt16,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_CompilerInt16T]],
) -> _CompilerInt16T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyUint16T,
    /,
    *,
    initial_value: _CutlassUint16,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_NumpyUint16T]],
) -> _NumpyUint16T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerUint16T,
    /,
    *,
    initial_value: _SeedUint16,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_CompilerUint16T]],
) -> _CompilerUint16T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyInt32T,
    /,
    *,
    initial_value: _CutlassInt32,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_NumpyInt32T]],
) -> _NumpyInt32T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerInt32T,
    /,
    *,
    initial_value: _SeedInt32,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_CompilerInt32T]],
) -> _CompilerInt32T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyUint32T,
    /,
    *,
    initial_value: _CutlassUint32,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_NumpyUint32T]],
) -> _NumpyUint32T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerUint32T,
    /,
    *,
    initial_value: _SeedUint32,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_CompilerUint32T]],
) -> _CompilerUint32T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyInt64T,
    /,
    *,
    initial_value: _CutlassInt64,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_NumpyInt64T]],
) -> _NumpyInt64T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerInt64T,
    /,
    *,
    initial_value: _SeedInt64,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_CompilerInt64T]],
) -> _CompilerInt64T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyUint64T,
    /,
    *,
    initial_value: _CutlassUint64,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_NumpyUint64T]],
) -> _NumpyUint64T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerUint64T,
    /,
    *,
    initial_value: _SeedUint64,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_CompilerUint64T]],
) -> _CompilerUint64T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyFloat32T,
    /,
    *,
    initial_value: _CutlassFloat32,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_NumpyFloat32T]],
) -> _NumpyFloat32T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerFloat32T,
    /,
    *,
    initial_value: _SeedFloat32,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_CompilerFloat32T]],
) -> _CompilerFloat32T: ...
@overload
def scan(
    group: WarpGroup,
    value: _NumpyFloat64T,
    /,
    *,
    initial_value: _CutlassFloat64,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_NumpyFloat64T]],
) -> _NumpyFloat64T: ...
@overload
def scan(
    group: WarpGroup,
    value: _CompilerFloat64T,
    /,
    *,
    initial_value: _SeedFloat64,
    **kwargs: Unpack[_WarpSeededScanModeOptions[_CompilerFloat64T]],
) -> _CompilerFloat64T: ...
@overload
def scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SeededScanOperator,
    initial_value: ContextualInitialValue[_ScalarT],
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _BuiltinScanOperator | None = None,
    initial_value: None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def exclusive_sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def exclusive_sum(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadData[_ItemT]: ...
@overload
def exclusive_sum(
    group: BlockGroup,
    value: _RegisterPayload,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[Any] | None = None,
) -> ThreadData[Any]: ...
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
) -> _ScalarT: ...
@overload
def inclusive_sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def inclusive_sum(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadData[_ItemT]: ...
@overload
def inclusive_sum(
    group: BlockGroup,
    value: _RegisterPayload,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[Any] | None = None,
) -> ThreadData[Any]: ...
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
) -> _ScalarT: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: SumScanOperator | _NumpySumScanUfunc | None = None,
    initial_value: ContextualInitialValue[_ScalarT] | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyInt8T,
    /,
    *,
    initial_value: _CutlassInt8,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyInt8T]],
) -> _NumpyInt8T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerInt8T,
    /,
    *,
    initial_value: _SeedInt8,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerInt8T]],
) -> _CompilerInt8T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyUint8T,
    /,
    *,
    initial_value: _CutlassUint8,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyUint8T]],
) -> _NumpyUint8T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerUint8T,
    /,
    *,
    initial_value: _SeedUint8,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerUint8T]],
) -> _CompilerUint8T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyInt16T,
    /,
    *,
    initial_value: _CutlassInt16,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyInt16T]],
) -> _NumpyInt16T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerInt16T,
    /,
    *,
    initial_value: _SeedInt16,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerInt16T]],
) -> _CompilerInt16T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyUint16T,
    /,
    *,
    initial_value: _CutlassUint16,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyUint16T]],
) -> _NumpyUint16T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerUint16T,
    /,
    *,
    initial_value: _SeedUint16,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerUint16T]],
) -> _CompilerUint16T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyInt32T,
    /,
    *,
    initial_value: _CutlassInt32,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyInt32T]],
) -> _NumpyInt32T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerInt32T,
    /,
    *,
    initial_value: _SeedInt32,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerInt32T]],
) -> _CompilerInt32T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyUint32T,
    /,
    *,
    initial_value: _CutlassUint32,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyUint32T]],
) -> _NumpyUint32T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerUint32T,
    /,
    *,
    initial_value: _SeedUint32,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerUint32T]],
) -> _CompilerUint32T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyInt64T,
    /,
    *,
    initial_value: _CutlassInt64,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyInt64T]],
) -> _NumpyInt64T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerInt64T,
    /,
    *,
    initial_value: _SeedInt64,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerInt64T]],
) -> _CompilerInt64T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyUint64T,
    /,
    *,
    initial_value: _CutlassUint64,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyUint64T]],
) -> _NumpyUint64T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerUint64T,
    /,
    *,
    initial_value: _SeedUint64,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerUint64T]],
) -> _CompilerUint64T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyFloat32T,
    /,
    *,
    initial_value: _CutlassFloat32,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyFloat32T]],
) -> _NumpyFloat32T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerFloat32T,
    /,
    *,
    initial_value: _SeedFloat32,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerFloat32T]],
) -> _CompilerFloat32T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _NumpyFloat64T,
    /,
    *,
    initial_value: _CutlassFloat64,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyFloat64T]],
) -> _NumpyFloat64T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _CompilerFloat64T,
    /,
    *,
    initial_value: _SeedFloat64,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerFloat64T]],
) -> _CompilerFloat64T: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _SeededScanOperator,
    initial_value: ContextualInitialValue[_ScalarT],
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    scan_op: SumScanOperator | _NumpySumScanUfunc | None = None,
    initial_value: ContextualInitialValue[_ItemT] | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadData[_ItemT]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt8T],
    /,
    *,
    initial_value: _CutlassInt8,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyInt8T]],
) -> ThreadData[_NumpyInt8T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt8T],
    /,
    *,
    initial_value: _SeedInt8,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerInt8T]],
) -> ThreadData[_CompilerInt8T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint8T],
    /,
    *,
    initial_value: _CutlassUint8,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyUint8T]],
) -> ThreadData[_NumpyUint8T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint8T],
    /,
    *,
    initial_value: _SeedUint8,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerUint8T]],
) -> ThreadData[_CompilerUint8T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt16T],
    /,
    *,
    initial_value: _CutlassInt16,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyInt16T]],
) -> ThreadData[_NumpyInt16T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt16T],
    /,
    *,
    initial_value: _SeedInt16,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerInt16T]],
) -> ThreadData[_CompilerInt16T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint16T],
    /,
    *,
    initial_value: _CutlassUint16,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyUint16T]],
) -> ThreadData[_NumpyUint16T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint16T],
    /,
    *,
    initial_value: _SeedUint16,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerUint16T]],
) -> ThreadData[_CompilerUint16T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt32T],
    /,
    *,
    initial_value: _CutlassInt32,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyInt32T]],
) -> ThreadData[_NumpyInt32T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt32T],
    /,
    *,
    initial_value: _SeedInt32,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerInt32T]],
) -> ThreadData[_CompilerInt32T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint32T],
    /,
    *,
    initial_value: _CutlassUint32,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyUint32T]],
) -> ThreadData[_NumpyUint32T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint32T],
    /,
    *,
    initial_value: _SeedUint32,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerUint32T]],
) -> ThreadData[_CompilerUint32T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyInt64T],
    /,
    *,
    initial_value: _CutlassInt64,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyInt64T]],
) -> ThreadData[_NumpyInt64T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerInt64T],
    /,
    *,
    initial_value: _SeedInt64,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerInt64T]],
) -> ThreadData[_CompilerInt64T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyUint64T],
    /,
    *,
    initial_value: _CutlassUint64,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyUint64T]],
) -> ThreadData[_NumpyUint64T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerUint64T],
    /,
    *,
    initial_value: _SeedUint64,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerUint64T]],
) -> ThreadData[_CompilerUint64T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyFloat32T],
    /,
    *,
    initial_value: _CutlassFloat32,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyFloat32T]],
) -> ThreadData[_NumpyFloat32T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerFloat32T],
    /,
    *,
    initial_value: _SeedFloat32,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerFloat32T]],
) -> ThreadData[_CompilerFloat32T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_NumpyFloat64T],
    /,
    *,
    initial_value: _CutlassFloat64,
    **kwargs: Unpack[_BlockSeededScanOptions[_NumpyFloat64T]],
) -> ThreadData[_NumpyFloat64T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_CompilerFloat64T],
    /,
    *,
    initial_value: _SeedFloat64,
    **kwargs: Unpack[_BlockSeededScanOptions[_CompilerFloat64T]],
) -> ThreadData[_CompilerFloat64T]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _RegisterPayload,
    /,
    *,
    scan_op: SumScanOperator | _NumpySumScanUfunc | None = None,
    initial_value: CommonNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[Any] | None = None,
) -> ThreadData[Any]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    scan_op: _SeededScanOperator,
    initial_value: ContextualInitialValue[_ItemT],
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadData[_ItemT]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _RegisterPayload,
    /,
    *,
    scan_op: _SeededScanOperator,
    initial_value: CommonNumericScalar,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[Any] | None = None,
) -> ThreadData[Any]: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: SumScanOperator | _NumpySumScanUfunc | None = None,
    initial_value: ContextualInitialValue[_ScalarT] | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyInt8T,
    /,
    *,
    initial_value: _CutlassInt8,
    **kwargs: Unpack[_WarpSeededScanOptions[_NumpyInt8T]],
) -> _NumpyInt8T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerInt8T,
    /,
    *,
    initial_value: _SeedInt8,
    **kwargs: Unpack[_WarpSeededScanOptions[_CompilerInt8T]],
) -> _CompilerInt8T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyUint8T,
    /,
    *,
    initial_value: _CutlassUint8,
    **kwargs: Unpack[_WarpSeededScanOptions[_NumpyUint8T]],
) -> _NumpyUint8T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerUint8T,
    /,
    *,
    initial_value: _SeedUint8,
    **kwargs: Unpack[_WarpSeededScanOptions[_CompilerUint8T]],
) -> _CompilerUint8T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyInt16T,
    /,
    *,
    initial_value: _CutlassInt16,
    **kwargs: Unpack[_WarpSeededScanOptions[_NumpyInt16T]],
) -> _NumpyInt16T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerInt16T,
    /,
    *,
    initial_value: _SeedInt16,
    **kwargs: Unpack[_WarpSeededScanOptions[_CompilerInt16T]],
) -> _CompilerInt16T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyUint16T,
    /,
    *,
    initial_value: _CutlassUint16,
    **kwargs: Unpack[_WarpSeededScanOptions[_NumpyUint16T]],
) -> _NumpyUint16T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerUint16T,
    /,
    *,
    initial_value: _SeedUint16,
    **kwargs: Unpack[_WarpSeededScanOptions[_CompilerUint16T]],
) -> _CompilerUint16T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyInt32T,
    /,
    *,
    initial_value: _CutlassInt32,
    **kwargs: Unpack[_WarpSeededScanOptions[_NumpyInt32T]],
) -> _NumpyInt32T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerInt32T,
    /,
    *,
    initial_value: _SeedInt32,
    **kwargs: Unpack[_WarpSeededScanOptions[_CompilerInt32T]],
) -> _CompilerInt32T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyUint32T,
    /,
    *,
    initial_value: _CutlassUint32,
    **kwargs: Unpack[_WarpSeededScanOptions[_NumpyUint32T]],
) -> _NumpyUint32T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerUint32T,
    /,
    *,
    initial_value: _SeedUint32,
    **kwargs: Unpack[_WarpSeededScanOptions[_CompilerUint32T]],
) -> _CompilerUint32T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyInt64T,
    /,
    *,
    initial_value: _CutlassInt64,
    **kwargs: Unpack[_WarpSeededScanOptions[_NumpyInt64T]],
) -> _NumpyInt64T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerInt64T,
    /,
    *,
    initial_value: _SeedInt64,
    **kwargs: Unpack[_WarpSeededScanOptions[_CompilerInt64T]],
) -> _CompilerInt64T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyUint64T,
    /,
    *,
    initial_value: _CutlassUint64,
    **kwargs: Unpack[_WarpSeededScanOptions[_NumpyUint64T]],
) -> _NumpyUint64T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerUint64T,
    /,
    *,
    initial_value: _SeedUint64,
    **kwargs: Unpack[_WarpSeededScanOptions[_CompilerUint64T]],
) -> _CompilerUint64T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyFloat32T,
    /,
    *,
    initial_value: _CutlassFloat32,
    **kwargs: Unpack[_WarpSeededScanOptions[_NumpyFloat32T]],
) -> _NumpyFloat32T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerFloat32T,
    /,
    *,
    initial_value: _SeedFloat32,
    **kwargs: Unpack[_WarpSeededScanOptions[_CompilerFloat32T]],
) -> _CompilerFloat32T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _NumpyFloat64T,
    /,
    *,
    initial_value: _CutlassFloat64,
    **kwargs: Unpack[_WarpSeededScanOptions[_NumpyFloat64T]],
) -> _NumpyFloat64T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _CompilerFloat64T,
    /,
    *,
    initial_value: _SeedFloat64,
    **kwargs: Unpack[_WarpSeededScanOptions[_CompilerFloat64T]],
) -> _CompilerFloat64T: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _SeededScanOperator,
    initial_value: ContextualInitialValue[_ScalarT],
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def inclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _BuiltinScanOperator | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def inclusive_scan(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    scan_op: _BuiltinScanOperator | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadData[_ItemT]: ...
@overload
def inclusive_scan(
    group: BlockGroup,
    value: _RegisterPayload,
    /,
    *,
    scan_op: _BuiltinScanOperator | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[Any] | None = None,
) -> ThreadData[Any]: ...
@overload
def inclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _BuiltinScanOperator | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
