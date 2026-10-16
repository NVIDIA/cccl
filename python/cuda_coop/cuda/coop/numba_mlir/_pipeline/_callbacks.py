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


def constant(ty) -> Any:
    """Read a static selector, preserving native callable types when present."""

    from ._templates import _omitted

    if _omitted(ty):
        return None
    if isinstance(ty, StaticCallableType):
        return ty.value
    if isinstance(ty, nb.types.Literal):
        return ty.literal_value
    if isinstance(ty, nb.types.Dispatcher):
        return ty.dispatcher
    if isinstance(ty, nb.types.Function):
        return ty.typing_key
    raise TypedPipelineError("cooperative selector must be static")


def register():
    nb.register_model(StaticCallableType)(_I8Model)
