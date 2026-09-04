# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stateful device-callable descriptor for qualified cooperative Scan."""

from typing import Generic

from typing_extensions import TypeVar

_OpT = TypeVar("_OpT")

class StatefulFunction(Generic[_OpT]):
    """Pair a device callable with its one-item state dtype."""

    op: _OpT
    dtype: object
    name: str | None

    def __init__(
        self,
        op: _OpT,
        dtype: object,
        name: str | None = None,
    ) -> None: ...

__all__ = ["StatefulFunction"]
