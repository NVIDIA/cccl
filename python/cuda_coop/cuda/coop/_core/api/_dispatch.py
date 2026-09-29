# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Register cooperative operations and their supported thread groups.

Compiler adapters use function identity to recognize calls such as load and
store, then check which group kinds the operation supports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from ..thread_group import ThreadGroup

_CallableT = TypeVar("_CallableT", bound=Callable[..., object])


@dataclass(frozen=True)
class _CommonGroupOperation:
    name: str
    group_kinds: tuple[str, ...]
    function: Callable[..., object]


_COMMON_GROUP_OPERATIONS_BY_NAME: dict[str, _CommonGroupOperation] = {}
_COMMON_GROUP_OPERATIONS_BY_FUNCTION: dict[
    Callable[..., object], _CommonGroupOperation
] = {}


def _common_group_operation(
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
            "common group operations require a name and group kinds"
        )

    def decorate(function: _CallableT) -> _CallableT:
        registration = _CommonGroupOperation(name, tuple(group_kinds), function)
        existing = _COMMON_GROUP_OPERATIONS_BY_NAME.get(name)
        if existing is not None and existing != registration:
            raise RuntimeError(
                f"common group operation {name!r} is already registered"
            )
        existing_function = _COMMON_GROUP_OPERATIONS_BY_FUNCTION.get(function)
        if existing_function is not None and existing_function != registration:
            raise RuntimeError(
                f"common group marker {function!r} is already registered"
            )
        _COMMON_GROUP_OPERATIONS_BY_NAME[name] = registration
        _COMMON_GROUP_OPERATIONS_BY_FUNCTION[function] = registration
        function.__cuda_coop_backend_member__ = name
        return function

    return decorate


def _common_group_operation_name(function: object) -> str | None:
    registration = _COMMON_GROUP_OPERATIONS_BY_FUNCTION.get(function)
    return None if registration is None else registration.name


class UnsupportedCoopBackendOperationError(NotImplementedError):
    """The selected compiler backend does not implement a root operation."""

    def __init__(self, backend_module: str, operation: str) -> None:
        self.backend_module = backend_module
        self.operation = operation
        self.reason_code = "cuda-coop-backend-operation-unavailable"
        super().__init__(
            f"cuda.coop.{operation} is not implemented by {backend_module!r}"
        )


def _common_group_name(kind: str) -> str:
    """Return the common API spelling for one internal group kind."""

    return "physical_warp" if kind == "warp" else kind


def _validate_common_operation_group(
    operation: str,
    group: object,
) -> None:
    """Reject groups outside an operation's common API contract.

    Compiler adapters validate a recognized common call after resolving its
    symbolic group and before lowering it. Backend-qualified operations may
    accept additional groups without changing this common API contract.
    """

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"cuda.coop.{operation} group must be a ThreadGroup")
    registration = _COMMON_GROUP_OPERATIONS_BY_NAME.get(operation)
    if registration is None:
        raise UnsupportedCoopBackendOperationError("cuda.coop", operation)
    supported = registration.group_kinds
    if group.kind in supported:
        return
    group_name = _common_group_name(group.kind)
    supported_names = ", ".join(map(_common_group_name, supported))
    raise NotImplementedError(
        f"cuda.coop.{operation} does not support group kind {group_name!r} in "
        f"the common API; supported group kinds: {supported_names}; use a "
        "backend-qualified import for backend-specific group support"
    )


__all__ = []
