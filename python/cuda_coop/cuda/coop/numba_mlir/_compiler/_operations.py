# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact callable identities recognized by the Numba-CUDA-MLIR planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])


@dataclass(frozen=True)
class FactoryOperation:
    """Semantic metadata for one planner-private lowering factory."""

    operation: str
    namespace: str


_GROUP_OPERATIONS: dict[Callable[..., Any], str] = {}
_FACTORY_OPERATIONS: dict[Callable[..., Any], FactoryOperation] = {}


def group_operation(operation: str) -> Callable[[_CallableT], _CallableT]:
    """Register a public marker by its exact callable identity."""

    def decorate(function: _CallableT) -> _CallableT:
        existing = _GROUP_OPERATIONS.get(function)
        if existing is not None and existing != operation:
            raise RuntimeError(
                f"group marker {function!r} is already registered as {existing!r}"
            )
        _GROUP_OPERATIONS[function] = operation
        function.__cuda_coop_backend_member__ = operation
        return function

    return decorate


def group_operation_name(function: Any) -> str | None:
    """Return the operation for an exactly registered public marker."""

    return _GROUP_OPERATIONS.get(function)


def register_factory(
    function: _CallableT,
    *,
    operation: str,
    namespace: str,
) -> _CallableT:
    """Register a lowering factory by exact callable identity."""

    metadata = FactoryOperation(operation=operation, namespace=namespace)
    existing = _FACTORY_OPERATIONS.get(function)
    if existing is not None and existing != metadata:
        raise RuntimeError(
            f"lowering factory {function!r} is already registered as {existing!r}"
        )
    _FACTORY_OPERATIONS[function] = metadata
    return function


def factory_operation(function: Any) -> FactoryOperation | None:
    """Return metadata for an exactly registered lowering factory."""

    return _FACTORY_OPERATIONS.get(function)


__all__ = [
    "FactoryOperation",
    "factory_operation",
    "group_operation",
    "group_operation_name",
    "register_factory",
]
