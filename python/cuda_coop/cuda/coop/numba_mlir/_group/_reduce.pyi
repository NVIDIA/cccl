# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Callable
from typing import Literal, Protocol, TypeAlias, overload

from typing_extensions import TypeVar

from ..._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ReduceAlgorithm,
    ReduceOperator,
    TempStorageLike,
    ValidItems,
)
from .._thread_group import BlockGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)
_ScalarT = TypeVar("_ScalarT", bound=CommonNumericScalar)

_NumpyReduceUfuncName: TypeAlias = Literal[
    "add",
    "multiply",
    "minimum",
    "maximum",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
]

class _NumpyReduceUfunc(Protocol):
    """Match supported NumPy ufunc aliases by name and arity.

    The literal name set and two-input, one-output arity identify built-in
    reduction aliases without relying on NumPy's ufunc stub types. Custom
    callbacks use separate callable overloads with CUB-only constraints.
    """

    @property
    def __name__(self) -> _NumpyReduceUfuncName: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[1]: ...

# Typeshed exposes ``operator.*`` functions as ``(Any, Any) -> Any``. An
# object-wide signature accepts those aliases while keeping dtype-specific
# custom callbacks on the custom callback overloads below.
_OperatorReduceAlias: TypeAlias = Callable[[object, object], object]
_BuiltinReduceOperator: TypeAlias = (
    ReduceOperator | _OperatorReduceAlias | _NumpyReduceUfunc
)
_CallbackReduceAlgorithm: TypeAlias = Literal["raking", "warp_reductions"]

@overload
def reduce(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    binary_op: _BuiltinReduceOperator | None = None,
    valid_items: None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT: ...
@overload
def reduce(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    binary_op: Callable[[_ItemT, _ItemT], _ItemT],
    valid_items: None = None,
    algorithm: _CallbackReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT: ...
@overload
def reduce(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: _BuiltinReduceOperator | None = None,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT: ...
@overload
def reduce(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: Callable[[_ScalarT, _ScalarT], _ScalarT],
    valid_items: ValidItems | None = None,
    algorithm: _CallbackReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT: ...
@overload
def reduce(
    group: WarpGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    binary_op: (
        _BuiltinReduceOperator | Callable[[_ItemT, _ItemT], _ItemT] | None
    ) = None,
    valid_items: None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ItemT: ...
@overload
def reduce(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: (
        _BuiltinReduceOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None
    ) = None,
    valid_items: ValidItems | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT: ...
@overload
def sum(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    valid_items: None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT: ...
@overload
def sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT: ...
@overload
def sum(
    group: WarpGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    valid_items: None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ItemT: ...
@overload
def sum(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    valid_items: ValidItems | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT: ...
