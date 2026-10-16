# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Static callback identities used only by cooperative specialization."""

from __future__ import annotations

from typing import Any

from . import _bridges as nb
from ._errors import TypedPipelineError
from ._types import _I8Model


class StaticCallableType(nb.types.Type):
    """A Python callable whose body the provider adapter compiles later."""

    def __init__(self, value):
        from .._semantic import _numba_semantic_token

        if not callable(value):
            raise TypedPipelineError("cooperative callback must be callable")
        self.value = value
        self.token = _numba_semantic_token(value)
        super().__init__(name=f"coop.callback[{self.token}]")

    @property
    def key(self):
        return self.token


class StatefulFunctionType(nb.types.Type):
    """A prefix callback descriptor, including its independent state dtype."""

    def __init__(self, value):
        from .._compiler._parameters import (
            _validate_common_numeric_dtype,
            normalize_dtype_param,
        )
        from .._semantic import _numba_semantic_token

        self.value = value
        self.dtype = _validate_common_numeric_dtype(
            normalize_dtype_param(value.dtype),
            operation="scan",
            parameter="prefix_state",
        )
        self.token = _numba_semantic_token(value.op)
        super().__init__(name=f"coop.prefix[{self.dtype}, {self.token}]")

    @property
    def key(self):
        return self.token, self.dtype


def constant(ty) -> Any:
    """Read a static selector, preserving native callable types when present."""

    from ._templates import _omitted

    if _omitted(ty):
        return None
    if isinstance(ty, (StaticCallableType, StatefulFunctionType)):
        return ty.value
    if isinstance(ty, nb.types.Literal):
        return ty.literal_value
    if isinstance(ty, nb.types.Dispatcher):
        return ty.dispatcher
    if isinstance(ty, nb.types.Function):
        return ty.typing_key
    raise TypedPipelineError("cooperative selector must be static")


def state_dtype(ty):
    return ty.dtype if isinstance(ty, StatefulFunctionType) else None


def register():
    from .._stateful_function import StatefulFunction

    nb.register_model(StaticCallableType)(_I8Model)
    nb.register_model(StatefulFunctionType)(_I8Model)
    nb.register_typeof(StatefulFunction)(
        lambda value, context: StatefulFunctionType(value)
    )
