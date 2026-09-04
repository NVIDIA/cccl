# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stateful device-callable descriptor for qualified cooperative Scan."""

from collections.abc import Callable
from typing import Generic, Protocol

from typing_extensions import TypeVar

from .._typing import PortableNumericScalar, ThreadDataLike

_StateT = TypeVar("_StateT", bound=PortableNumericScalar)
_ValueT = TypeVar("_ValueT", bound=PortableNumericScalar)

class _StatefulFunctor(Protocol[_ValueT]):
    def __call__(self, block_aggregate: _ValueT, /) -> _ValueT: ...

class StatefulFunction(Generic[_StateT, _ValueT]):
    """Pair a device callable with its one-item state dtype."""

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
