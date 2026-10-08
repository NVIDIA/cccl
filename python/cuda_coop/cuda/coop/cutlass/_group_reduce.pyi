# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe qualified reduction shapes and root-only CUB options.

Overloads preserve the input element type and separate full-group payloads
from scalar valid prefixes. Structural callable types admit built-in aliases;
tracing checks their identities and rejects custom callbacks.
"""

from collections.abc import Callable
from typing import Literal, Protocol, TypeAlias, overload

from typing_extensions import TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ReduceAlgorithm,
    ReduceOperator,
    TempStorageLike,
    ValidItems,
)

from .._core.api.thread_group import BlockGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)

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
    """Describe NumPy binary ufunc metadata for built-in alias typing."""

    @property
    def __name__(self) -> _NumpyReduceUfuncName: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[1]: ...

# Built-in operator aliases have object-wide types. The compiler validates
# known callable identities; arbitrary user callbacks are unsupported.
_OperatorReduceAlias: TypeAlias = Callable[[object, object], object]
_BuiltinReduceOperator: TypeAlias = (
    ReduceOperator | _OperatorReduceAlias | _NumpyReduceUfunc
)

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
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def reduce(
    group: BlockGroup,
    value: _ItemT,
    /,
    *,
    binary_op: _BuiltinReduceOperator | None = None,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def reduce(
    group: WarpGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    binary_op: _BuiltinReduceOperator | None = None,
    valid_items: None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def reduce(
    group: WarpGroup,
    value: _ItemT,
    /,
    *,
    binary_op: _BuiltinReduceOperator | None = None,
    valid_items: ValidItems | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def sum(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    valid_items: None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def sum(
    group: BlockGroup,
    value: _ItemT,
    /,
    *,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def sum(
    group: WarpGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    valid_items: None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""

@overload
def sum(
    group: WarpGroup,
    value: _ItemT,
    /,
    *,
    valid_items: ValidItems | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ItemT:
    """Return a reduced scalar defined only at group rank zero."""
