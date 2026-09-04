# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scan signatures for block and warp groups."""

from collections.abc import Callable
from typing import Literal, Protocol, TypeAlias, overload

from typing_extensions import TypeVar

from .._typing import (
    NonSumScanOperator,
    PortableNumericScalar,
    ScanAlgorithm,
    ScanOperator,
    SumScanOperator,
    ThreadDataLike,
    ValidItems,
)
from ._enums import BlockScanAlgorithm
from ._temp_storage import TempStorage
from ._thread_group import BlockGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=PortableNumericScalar)
_ScalarT = TypeVar("_ScalarT", bound=PortableNumericScalar)
_ContextualInitial: TypeAlias = int | float

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

_OperatorScanAlias: TypeAlias = Callable[[object, object], object]
_ItemScanCallable: TypeAlias = Callable[[_ItemT, _ItemT], _ItemT]
_ScalarScanCallable: TypeAlias = Callable[[_ScalarT, _ScalarT], _ScalarT]
_KnownItemScanOperator: TypeAlias = (
    ScanOperator | _OperatorScanAlias | _NumpyScanUfunc | _ItemScanCallable[_ItemT]
)
_KnownScalarScanOperator: TypeAlias = (
    ScanOperator | _OperatorScanAlias | _NumpyScanUfunc | _ScalarScanCallable[_ScalarT]
)
_NonSumItemScanOperator: TypeAlias = (
    NonSumScanOperator
    | _OperatorScanAlias
    | _NumpyScanUfunc
    | _ItemScanCallable[_ItemT]
)
_NonSumScalarScanOperator: TypeAlias = (
    NonSumScanOperator
    | _OperatorScanAlias
    | _NumpyScanUfunc
    | _ScalarScanCallable[_ScalarT]
)
_BlockAlgorithm: TypeAlias = ScanAlgorithm | BlockScanAlgorithm

@overload
def scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: _ItemT | _ContextualInitial | None = None,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumItemScanOperator[_ItemT],
    initial_value: _ItemT | _ContextualInitial,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _KnownItemScanOperator[_ItemT] | None = None,
    initial_value: None = None,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: _ScalarT | _ContextualInitial | None = None,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
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
    scan_op: _NonSumScalarScanOperator[_ScalarT],
    initial_value: _ScalarT | _ContextualInitial,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
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
    scan_op: _KnownScalarScanOperator[_ScalarT] | None = None,
    initial_value: None = None,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: SumScanOperator | None = None,
    initial_value: _ScalarT | _ContextualInitial | None = None,
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
    scan_op: _NonSumScalarScanOperator[_ScalarT],
    initial_value: _ScalarT | _ContextualInitial,
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
    scan_op: _KnownScalarScanOperator[_ScalarT] | None = None,
    initial_value: None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: _ItemT | _ContextualInitial | None = None,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: _NonSumItemScanOperator[_ItemT],
    initial_value: _ItemT | _ContextualInitial,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: _ScalarT | _ContextualInitial | None = None,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def exclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _NonSumScalarScanOperator[_ScalarT],
    initial_value: _ScalarT | _ContextualInitial,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def exclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: SumScanOperator | None = None,
    initial_value: _ScalarT | _ContextualInitial | None = None,
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
    scan_op: _NonSumScalarScanOperator[_ScalarT],
    initial_value: _ScalarT | _ContextualInitial,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def inclusive_scan(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: _KnownItemScanOperator[_ItemT] | None = None,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def inclusive_scan(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _KnownScalarScanOperator[_ScalarT] | None = None,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def inclusive_scan(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _KnownScalarScanOperator[_ScalarT] | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
@overload
def exclusive_sum(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def exclusive_sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
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
    value: ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ItemT] | None = None,
) -> ThreadDataLike[_ItemT]: ...
@overload
def inclusive_sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _BlockAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT: ...
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
