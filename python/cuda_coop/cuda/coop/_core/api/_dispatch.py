# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable root-backend selection and dispatch state.

Compiler integrations activate one qualified backend through this module while
family frontends validate the portable profile before delegation. It owns no
primitive semantics, provider rendering, or compiler cache state.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from types import ModuleType
from typing import Any

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup

_ACTIVE_BACKEND_MODULE: ContextVar[str | None] = ContextVar(
    "cuda_coop_active_backend_module",
    default=None,
)
_QUALIFIED_BACKEND_MODULE: str | None = None
_ACTIVE_COMMON_ROOT_OPERATION: ContextVar[str | None] = ContextVar(
    "cuda_coop_active_common_root_operation",
    default=None,
)
_GROUP_OPERATIONS = (
    "adjacent_difference",
    "discontinuity",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "histogram",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "reduce",
    "run_length_decode",
    "scan",
    "shuffle",
    "store",
    "sum",
)

_BLOCK_AND_WARP_GROUPS = ("block", "warp", "threads_within_warp")
_REDUCTION_GROUPS = (
    "thread",
    "warp",
    "threads_within_warp",
    "block",
    "cluster",
)
_BLOCK_ONLY = ("block",)
_PORTABLE_OPERATION_GROUPS = {
    "load": _BLOCK_AND_WARP_GROUPS,
    "store": _BLOCK_AND_WARP_GROUPS,
    "reduce": _REDUCTION_GROUPS,
    "sum": _REDUCTION_GROUPS,
    "scan": _BLOCK_AND_WARP_GROUPS,
    "exclusive_sum": _BLOCK_AND_WARP_GROUPS,
    "inclusive_sum": _BLOCK_AND_WARP_GROUPS,
    "exclusive_scan": _BLOCK_AND_WARP_GROUPS,
    "inclusive_scan": _BLOCK_AND_WARP_GROUPS,
    "exchange": _BLOCK_AND_WARP_GROUPS,
    "adjacent_difference": _BLOCK_ONLY,
    "discontinuity": _BLOCK_ONLY,
    "shuffle": _BLOCK_ONLY,
    "histogram": _BLOCK_ONLY,
    "run_length_decode": _BLOCK_ONLY,
}
_LOAD_STORE_ALGORITHMS = frozenset(
    {
        "direct",
        "striped",
        "vectorize",
        "transpose",
        "warp_transpose",
        "warp_transpose_timesliced",
    }
)
_REDUCE_ALGORITHMS = frozenset({"raking_commutative_only", "raking", "warp_reductions"})
_SCAN_ALGORITHMS = frozenset({"raking", "raking_memoize", "warp_scans"})
_SCAN_MODES = frozenset({"exclusive", "inclusive"})
_PORTABLE_OPERATOR_ALIASES = {
    "+": "sum",
    "sum": "sum",
    "add": "sum",
    "plus": "sum",
    "*": "multiplies",
    "mul": "multiplies",
    "multiply": "multiplies",
    "multiplies": "multiplies",
    "min": "min",
    "minimum": "min",
    "max": "max",
    "maximum": "max",
    "&": "bit_and",
    "bit_and": "bit_and",
    "|": "bit_or",
    "bit_or": "bit_or",
    "^": "bit_xor",
    "bit_xor": "bit_xor",
}
_EXCHANGE_MODES = frozenset({"striped_to_blocked", "blocked_to_striped"})
_ADJACENT_DIFFERENCE_DIRECTIONS = frozenset({"left", "right"})
_DISCONTINUITY_MODES = frozenset({"heads", "tails"})
_SHUFFLE_MODES = frozenset({"down", "up"})
_HISTOGRAM_ALGORITHMS = frozenset({"atomic", "sort"})


class UnsupportedCoopBackendOperationError(NotImplementedError):
    """The selected compiler backend does not implement a root operation."""

    def __init__(self, backend_module: str, operation: str) -> None:
        self.backend_module = backend_module
        self.operation = operation
        self.reason_code = "cuda-coop-backend-operation-unavailable"
        super().__init__(
            f"cuda.coop.{operation} is not implemented by {backend_module!r}"
        )


@contextmanager
def _compiler_scope(backend_module: str) -> Iterator[None]:
    """Activate one backend for the current compiler trace."""

    if not isinstance(backend_module, str):
        raise TypeError("backend_module must be a string")
    if not backend_module.strip():
        raise ValueError("backend_module must be a non-empty string")

    token = _ACTIVE_BACKEND_MODULE.set(backend_module)
    try:
        yield
    finally:
        _ACTIVE_BACKEND_MODULE.reset(token)


def _register_qualified_backend(backend_module: str) -> None:
    """Use one imported qualified backend outside compiler-owned scopes."""

    if not isinstance(backend_module, str):
        raise TypeError("backend_module must be a string")
    if not backend_module.strip():
        raise ValueError("backend_module must be a non-empty string")

    global _QUALIFIED_BACKEND_MODULE
    if _QUALIFIED_BACKEND_MODULE not in {None, backend_module}:
        raise RuntimeError(
            "cuda.coop common-root fallback is already activated by "
            f"{_QUALIFIED_BACKEND_MODULE!r}; cannot also activate "
            f"{backend_module!r}"
        )
    _QUALIFIED_BACKEND_MODULE = backend_module


def _backend_module_name() -> str | None:
    return _ACTIVE_BACKEND_MODULE.get() or _QUALIFIED_BACKEND_MODULE


@contextmanager
def _common_root_operation_scope(operation: str) -> Iterator[None]:
    """Identify one root dispatch without changing backend selection."""

    token = _ACTIVE_COMMON_ROOT_OPERATION.set(operation)
    try:
        yield
    finally:
        _ACTIVE_COMMON_ROOT_OPERATION.reset(token)


def _common_root_operation_name() -> str | None:
    """Return the common-root operation currently delegated to a backend."""

    return _ACTIVE_COMMON_ROOT_OPERATION.get()


def _active_backend(feature: str) -> tuple[str, ModuleType]:
    module_name = _backend_module_name()
    if module_name is None:
        raise CoopCompilerContextRequiredError(
            f"cuda.coop.{feature} requires compiler-owned activation or a "
            "qualified backend import before compilation"
        )
    return module_name, import_module(module_name)


def _backend_member(name: str) -> Any:
    module_name, backend = _active_backend(name)
    try:
        return getattr(backend, name)
    except AttributeError as exc:
        raise UnsupportedCoopBackendOperationError(module_name, name) from exc


def _portable_selector(
    operation: str,
    parameter: str,
    value: Any,
    allowed: frozenset[str],
    *,
    allow_none: bool = False,
) -> Any:
    """Normalize one common selector while a tracing backend is active."""

    if _backend_module_name() is None:
        return value
    if value is None and allow_none:
        return None
    token = getattr(value, "value", value)
    if isinstance(token, str):
        token = token.strip().lower().replace("-", "_")
    try:
        is_allowed = token in allowed
    except TypeError:
        is_allowed = False
    if not is_allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(
            f"cuda.coop.{operation} {parameter} must be one of: {choices}; "
            "use a backend-qualified import for backend-only controls"
        )
    return token


def _portable_operator(operation: str, parameter: str, value: Any) -> Any:
    """Normalize one built-in operator in the common portable profile."""

    if _backend_module_name() is None or value is None:
        return value
    token = getattr(value, "value", value)
    if isinstance(token, str):
        token = token.strip().lower().replace("-", "_")
    try:
        return _PORTABLE_OPERATOR_ALIASES[token]
    except (KeyError, TypeError):
        choices = ", ".join(sorted(set(_PORTABLE_OPERATOR_ALIASES.values())))
        raise ValueError(
            f"cuda.coop.{operation} {parameter} must be one of: {choices}; "
            "use a backend-qualified import for backend-only operators"
        ) from None


def _portable_group_name(kind: str) -> str:
    """Return the portable API spelling for one internal group kind."""

    return "physical_warp" if kind == "warp" else kind


def _validate_portable_operation_group(
    operation: str,
    group: Any,
) -> None:
    """Enforce the portable group matrix for a common-root call."""

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"cuda.coop.{operation} group must be a ThreadGroup")
    supported = _PORTABLE_OPERATION_GROUPS[operation]
    if group.kind in supported:
        return
    group_name = _portable_group_name(group.kind)
    supported_names = ", ".join(map(_portable_group_name, supported))
    raise NotImplementedError(
        f"cuda.coop.{operation} does not support group kind {group_name!r} in "
        f"the portable API; supported group kinds: {supported_names}; use a "
        "backend-qualified import for backend-specific group support"
    )


def _group_primitive_marker(
    operation: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if _backend_module_name() is None:
        del args, kwargs
        raise CoopCompilerContextRequiredError(
            f"cuda.coop.{operation} requires compiler-owned activation or a "
            "qualified backend import before compilation"
        )
    _validate_portable_operation_group(operation, args[0] if args else None)
    with _common_root_operation_scope(operation):
        return _backend_member(operation)(*args, **kwargs)


__all__ = []
