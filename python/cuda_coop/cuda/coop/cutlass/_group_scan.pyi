# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Built-in Scan signatures for the qualified CUTLASS backend."""

from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeAlias, overload

from typing_extensions import TypeVar

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
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)
_ScalarT = TypeVar("_ScalarT", bound=CommonNumericScalar)
_RegisterPayload: TypeAlias = CutlassTensorSample | CutlassTensorSSASample
_NumpyScanUfuncName: TypeAlias = Literal[
    "add", "multiply", "minimum", "maximum", "bitwise_and", "bitwise_or", "bitwise_xor"
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
_BuiltinScanOperator: TypeAlias = ScanOperator | _OperatorScanAlias | _NumpyScanUfunc
_SeededScanOperator: TypeAlias = (
    NonSumScanOperator | _OperatorScanAlias | _NumpyScanUfunc
)

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
