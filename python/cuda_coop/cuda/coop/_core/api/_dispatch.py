# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Index cooperative calls by function identity and supported group kinds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

_CallableT = TypeVar("_CallableT", bound=Callable[..., object])


@dataclass(frozen=True)
class _PortableGroupOperation:
    name: str
    group_kinds: tuple[str, ...]
    function: Callable[..., object]


_PORTABLE_GROUP_OPERATIONS_BY_NAME: dict[str, _PortableGroupOperation] = {}
_PORTABLE_GROUP_OPERATIONS_BY_FUNCTION: dict[
    Callable[..., object], _PortableGroupOperation
] = {}


def _portable_group_operation(
    name: str,
    *,
    group_kinds: tuple[str, ...],
) -> Callable[[_CallableT], _CallableT]:
    """Record a public operation and the groups it accepts.

    The decorator runs when the API module is imported and records the
    function object without wrapping it. Compiler adapters use that object
    to recognize calls, including imported aliases. An unrelated function
    with the same name does not acquire this registration.
    """

    if not name or not group_kinds:
        raise ValueError(
            "portable group operations require a name and group kinds"
        )

    def decorate(function: _CallableT) -> _CallableT:
        registration = _PortableGroupOperation(
            name, tuple(group_kinds), function
        )
        existing = _PORTABLE_GROUP_OPERATIONS_BY_NAME.get(name)
        if existing is not None and existing != registration:
            raise RuntimeError(
                f"portable group operation {name!r} is already registered"
            )
        existing_function = _PORTABLE_GROUP_OPERATIONS_BY_FUNCTION.get(function)
        if existing_function is not None and existing_function != registration:
            raise RuntimeError(
                f"portable group marker {function!r} is already registered"
            )
        _PORTABLE_GROUP_OPERATIONS_BY_NAME[name] = registration
        _PORTABLE_GROUP_OPERATIONS_BY_FUNCTION[function] = registration
        function.__cuda_coop_backend_member__ = name
        return function

    return decorate


__all__ = []
