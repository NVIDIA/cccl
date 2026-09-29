# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Index cooperative calls by function identity and supported group kinds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from ..thread_group import ThreadGroup

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
    """Register one common group overload by exact callable identity."""

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


def _portable_group_operation_name(function: object) -> str | None:
    registration = _PORTABLE_GROUP_OPERATIONS_BY_FUNCTION.get(function)
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


def _portable_group_name(kind: str) -> str:
    """Return the common API spelling for one internal group kind."""

    return "physical_warp" if kind == "warp" else kind


def _validate_portable_operation_group(
    operation: str,
    group: object,
) -> None:
    """Enforce the common group matrix for a common-root call."""

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"cuda.coop.{operation} group must be a ThreadGroup")
    registration = _PORTABLE_GROUP_OPERATIONS_BY_NAME.get(operation)
    if registration is None:
        raise UnsupportedCoopBackendOperationError("cuda.coop", operation)
    supported = registration.group_kinds
    if group.kind in supported:
        return
    group_name = _portable_group_name(group.kind)
    supported_names = ", ".join(map(_portable_group_name, supported))
    raise NotImplementedError(
        f"cuda.coop.{operation} does not support group kind {group_name!r} in "
        f"the portable API; supported group kinds: {supported_names}; use a "
        "backend-qualified import for backend-specific group support"
    )


__all__ = []
