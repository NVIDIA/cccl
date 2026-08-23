# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stable semantic identities and internal symbol helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import re
from enum import Enum
from functools import partial
from types import CodeType, ModuleType
from typing import Any, Mapping

_ADDRESS_IN_REPR = re.compile(r"(?<= at )0x[0-9a-fA-F]+")


def internal_mangle_cpp(cpp_name: str) -> str:
    """Mangle a C++ spelling into an internal identifier component."""

    return re.sub(r"[^a-zA-Z0-9]", "_", cpp_name)


def mangle_symbol(name: str, components: tuple[Any, ...]) -> str:
    """Combine a base name with already rendered identifier components."""

    return "_".join([name, *(internal_mangle_cpp(str(value)) for value in components)])


def _code_token(code: CodeType) -> tuple[Any, ...]:
    constants = tuple(
        _code_token(value) if isinstance(value, CodeType) else semantic_token(value)
        for value in code.co_consts
    )
    return (
        code.co_code.hex(),
        constants,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
    )


def _object_state_token(value: Any) -> tuple[Any, ...] | None:
    state: list[tuple[str, Any]] = []
    attributes = getattr(value, "__dict__", None)
    if attributes:
        state.append(("__dict__", semantic_token(attributes)))

    seen_slots: set[str] = set()
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in {"__dict__", "__weakref__"} or slot in seen_slots:
                continue
            seen_slots.add(slot)
            try:
                slot_value = getattr(value, slot)
            except AttributeError:
                continue
            token = ("self",) if slot_value is value else semantic_token(slot_value)
            state.append((slot, token))

    return tuple(state) if state else None


def _callable_token(value: Any) -> tuple[Any, ...]:
    if isinstance(value, partial):
        return (
            "partial",
            semantic_token(value.func),
            semantic_token(value.args),
            semantic_token(value.keywords),
        )

    code = getattr(value, "__code__", None)
    callable_state = None
    if code is None:
        call = getattr(value, "__call__", None)
        code = getattr(call, "__code__", None)
        callable_state = _object_state_token(value)

    digest = hashlib.sha256()
    if code is not None:
        digest.update(
            repr(_code_token(code)).encode("utf-8", errors="backslashreplace")
        )
    digest.update(
        repr(semantic_token(getattr(value, "__defaults__", None))).encode(
            "utf-8", errors="backslashreplace"
        )
    )
    digest.update(
        repr(semantic_token(getattr(value, "__kwdefaults__", None))).encode(
            "utf-8", errors="backslashreplace"
        )
    )
    closure = getattr(value, "__closure__", None) or ()
    for cell in closure:
        try:
            cell_value = cell.cell_contents
        except ValueError:
            cell_value = "<empty>"
        digest.update(
            repr(semantic_token(cell_value)).encode("utf-8", errors="backslashreplace")
        )
    bound_self = getattr(value, "__self__", None)
    if bound_self is not None:
        if isinstance(bound_self, ModuleType):
            bound_self_token = ("module", bound_self.__name__)
        else:
            state = _object_state_token(bound_self)
            bound_self_token = (
                type(bound_self).__module__,
                type(bound_self).__qualname__,
                state,
            )
        digest.update(repr(bound_self_token).encode("utf-8", errors="backslashreplace"))
    if callable_state is not None:
        digest.update(repr(callable_state).encode("utf-8", errors="backslashreplace"))

    return (
        "callable",
        getattr(value, "__module__", type(value).__module__),
        getattr(value, "__qualname__", type(value).__qualname__),
        digest.hexdigest()[:20],
    )


def semantic_token(value: Any) -> Any:
    """Return a deterministic, hashable description of a semantic value."""

    if isinstance(value, Enum):
        return type(value).__module__, type(value).__qualname__, value.value
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, type):
        return "type", value.__module__, value.__qualname__
    if isinstance(value, Mapping):
        items = [
            (semantic_token(key), semantic_token(item)) for key, item in value.items()
        ]
        return tuple(sorted(items, key=lambda item: repr(item[0])))
    if isinstance(value, (tuple, list)):
        return tuple(semantic_token(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((semantic_token(item) for item in value), key=repr))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (field.name, semantic_token(getattr(value, field.name)))
                for field in dataclasses.fields(value)
            ),
        )
    if callable(value):
        return _callable_token(value)

    attributes = _object_state_token(value)
    if attributes is not None:
        return (
            type(value).__module__,
            type(value).__qualname__,
            attributes,
        )
    stable_repr = _ADDRESS_IN_REPR.sub("0x?", repr(value))
    return type(value).__module__, type(value).__qualname__, stable_repr
