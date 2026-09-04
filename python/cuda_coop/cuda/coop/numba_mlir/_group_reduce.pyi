# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reduction signatures for supported thread groups."""

from collections.abc import Callable
from typing import Literal, Protocol, TypeAlias, overload

from typing_extensions import TypeVar

from .._typing import (
    PortableNumericScalar,
    ReduceAlgorithm,
    ReduceOperator,
    ThreadDataLike,
    ValidItems,
)
from ._thread_group import BlockGroup, ReductionGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=PortableNumericScalar)
_ScalarT = TypeVar("_ScalarT", bound=PortableNumericScalar)

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
    @property
    def __name__(self) -> _NumpyReduceUfuncName: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[1]: ...

# Typeshed exposes ``operator.*`` functions as ``(Any, Any) -> Any``. An
# object-wide signature accepts those aliases while keeping dtype-specific
# custom callbacks on the CUB-only, ``broadcast=False`` overloads below.
_OperatorReduceAlias: TypeAlias = Callable[[object, object], object]
_CudaxReduceOperator: TypeAlias = (
    ReduceOperator | _OperatorReduceAlias | _NumpyReduceUfunc
)
_CallbackReduceAlgorithm: TypeAlias = Literal["raking", "warp_reductions"]

@overload
def reduce(
    group: ReductionGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    binary_op: _CudaxReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ItemT: ...
@overload
def reduce(
    group: ReductionGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: _CudaxReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ScalarT: ...
@overload
def reduce(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    binary_op: _CudaxReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm | None = None,
) -> _ItemT: ...
@overload
def reduce(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    binary_op: Callable[[_ItemT, _ItemT], _ItemT],
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: _CallbackReduceAlgorithm | None = None,
) -> _ItemT: ...
@overload
def reduce(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: _CudaxReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
) -> _ScalarT: ...
@overload
def reduce(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: Callable[[_ScalarT, _ScalarT], _ScalarT],
    broadcast: Literal[False],
    valid_items: ValidItems | None = None,
    algorithm: _CallbackReduceAlgorithm | None = None,
) -> _ScalarT: ...
@overload
def reduce(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: (
        _CudaxReduceOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None
    ) = None,
    broadcast: Literal[False],
    valid_items: ValidItems | None = None,
    algorithm: None = None,
) -> _ScalarT: ...
@overload
def sum(
    group: ReductionGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ItemT: ...
@overload
def sum(
    group: ReductionGroup,
    value: _ScalarT,
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ScalarT: ...
@overload
def sum(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: ReduceAlgorithm,
) -> _ItemT: ...
@overload
def sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
) -> _ScalarT: ...
@overload
def sum(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: ValidItems,
    algorithm: None = None,
) -> _ScalarT: ...
