# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type a Scan callback independently of its mutable state payload.

The two type parameters distinguish the state element from the scanned
value. Overloads require a matching state array when this descriptor is
used, while the callback returns the same type as its aggregate input.
"""

from collections.abc import Callable
from typing import Generic, Protocol

from typing_extensions import TypeVar

from .._typing import CommonNumericScalar, ThreadDataLike

_StateT = TypeVar("_StateT", bound=CommonNumericScalar)
_ValueT = TypeVar("_ValueT", bound=CommonNumericScalar)

class _StatefulFunctor(Protocol[_ValueT]):
    """Describe the class form accepted for a stateful prefix callback.

    The compiler uses the class's unbound ``__call__`` method. Its first
    parameter receives the device state pointer in place of an instance;
    this protocol records the aggregate argument and return type.
    """

    def __call__(self, block_aggregate: _ValueT, /) -> _ValueT: ...

class StatefulFunction(Generic[_StateT, _ValueT]):
    """Pair a device callback with an independent one-item state dtype.

    ``_StateT`` describes the mutable state element; ``_ValueT`` describes
    both the aggregate input and returned prefix. The implementation's
    public docstring specifies initialization and synchronization rules.
    """

    op: (
        Callable[[ThreadDataLike[_StateT], _ValueT], _ValueT]
        | type[_StatefulFunctor[_ValueT]]
    )
    dtype: object
    name: str | None

    def __init__(
        self,
        op: (
            Callable[[ThreadDataLike[_StateT], _ValueT], _ValueT]
            | type[_StatefulFunctor[_ValueT]]
        ),
        dtype: object,
        name: str | None = None,
    ) -> None: ...

__all__ = ["StatefulFunction"]
