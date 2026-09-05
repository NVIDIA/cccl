# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stable semantic identities and internal symbol helpers."""

from __future__ import annotations

import dataclasses
import dis
import hashlib
import math
import re
from collections import defaultdict
from enum import Enum
from functools import partial
from types import CodeType, ModuleType
from typing import Any, Mapping

_ADDRESS_IN_REPR = re.compile(r"(?<= at )0x[0-9a-fA-F]+")


@dataclasses.dataclass
class _TokenState:
    active: dict[int, int] = dataclasses.field(default_factory=dict)


def _code_token(code: CodeType, state: _TokenState) -> tuple[Any, ...]:
    constants = tuple(
        (
            _code_token(value, state)
            if isinstance(value, CodeType)
            else _semantic_token(value, state)
        )
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


def _object_state_token(
    value: Any,
    state: _TokenState,
) -> tuple[Any, ...] | None:
    object_state: list[tuple[str, Any]] = []
    attributes = getattr(value, "__dict__", None)
    if attributes:
        object_state.append(("__dict__", _semantic_token(attributes, state)))

    seen_slots: set[str] = set()
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            storage_name = slot
            if slot.startswith("__") and not slot.endswith("__"):
                class_name = cls.__name__.lstrip("_")
                if class_name:
                    storage_name = f"_{class_name}{slot}"
            if storage_name in {"__dict__", "__weakref__"} or (
                storage_name in seen_slots
            ):
                continue
            seen_slots.add(storage_name)
            try:
                slot_value = getattr(value, storage_name)
            except AttributeError:
                continue
            token = (
                ("self",) if slot_value is value else _semantic_token(slot_value, state)
            )
            object_state.append((storage_name, token))

    return tuple(object_state) if object_state else None


def _referenced_global_names(code: CodeType) -> tuple[str, ...]:
    names = {
        instruction.argval
        for instruction in dis.get_instructions(code)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
        and isinstance(instruction.argval, str)
    }
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names.update(_referenced_global_names(constant))
    return tuple(sorted(names))


def _callable_token(value: Any, state: _TokenState) -> tuple[Any, ...]:
    if isinstance(value, partial):
        return (
            "partial",
            _semantic_token(value.func, state),
            _semantic_token(value.args, state),
            _semantic_token(value.keywords, state),
        )

    code = getattr(value, "__code__", None)
    global_namespace = getattr(value, "__globals__", None)
    callable_definition = value
    callable_state = None
    if code is None:
        call = getattr(value, "__call__", None)
        code = getattr(call, "__code__", None)
        global_namespace = getattr(call, "__globals__", None)
        callable_definition = call
        callable_state = _object_state_token(value, state)

    digest = hashlib.sha256()
    if code is not None:
        digest.update(
            repr(_code_token(code, state)).encode("utf-8", errors="backslashreplace")
        )
        if isinstance(global_namespace, Mapping):
            global_tokens = []
            for name in _referenced_global_names(code):
                if name not in global_namespace:
                    continue
                global_value = global_namespace[name]
                token = (
                    ("module", global_value.__name__)
                    if isinstance(global_value, ModuleType)
                    else _semantic_token(global_value, state)
                )
                global_tokens.append((name, token))
            digest.update(
                repr(tuple(global_tokens)).encode("utf-8", errors="backslashreplace")
            )
    digest.update(
        repr(
            _semantic_token(
                getattr(callable_definition, "__defaults__", None),
                state,
            )
        ).encode("utf-8", errors="backslashreplace")
    )
    digest.update(
        repr(
            _semantic_token(
                getattr(callable_definition, "__kwdefaults__", None),
                state,
            )
        ).encode("utf-8", errors="backslashreplace")
    )
    closure = getattr(callable_definition, "__closure__", None) or ()
    for cell in closure:
        try:
            cell_value = cell.cell_contents
        except ValueError:
            cell_value = "<empty>"
        digest.update(
            repr(_semantic_token(cell_value, state)).encode(
                "utf-8", errors="backslashreplace"
            )
        )
    bound_self = getattr(value, "__self__", None)
    if bound_self is not None:
        if isinstance(bound_self, ModuleType):
            bound_self_token = ("module", bound_self.__name__)
        else:
            object_state = _object_state_token(bound_self, state)
            bound_self_token = (
                type(bound_self).__module__,
                type(bound_self).__qualname__,
                object_state,
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


def _cycle_token(value: Any, back_reference_depth: int) -> tuple[str, str, str, int]:
    return (
        "cycle",
        getattr(value, "__module__", type(value).__module__),
        getattr(value, "__qualname__", type(value).__qualname__),
        back_reference_depth,
    )


def _container_kind(value: Any) -> tuple[str, str]:
    return type(value).__module__, type(value).__qualname__


def _container_state_token(
    value: Any,
    state: _TokenState,
) -> tuple[tuple[str, Any], ...] | None:
    container_state = list(_object_state_token(value, state) or ())
    if isinstance(value, defaultdict):
        default_factory = value.default_factory
        default_factory_token = (
            ("self",)
            if default_factory is value
            else _semantic_token(default_factory, state)
        )
        container_state.append(("default_factory", default_factory_token))
    return tuple(container_state) if container_state else None


def _semantic_token(value: Any, state: _TokenState) -> Any:
    if isinstance(value, Enum):
        return type(value).__module__, type(value).__qualname__, value.value
    if isinstance(value, float):
        if math.isnan(value):
            return "float", "nan"
        return "float", value.hex()
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, type):
        return "type", value.__module__, value.__qualname__

    value_id = id(value)
    if value_id in state.active:
        return _cycle_token(value, len(state.active) - state.active[value_id])
    state.active[value_id] = len(state.active)
    try:
        if isinstance(value, Mapping):
            items = [
                (
                    _semantic_token(key, state),
                    _semantic_token(item, state),
                )
                for key, item in value.items()
            ]
            return (
                "mapping",
                _container_kind(value),
                _container_state_token(value, state),
                tuple(sorted(items, key=lambda item: repr(item[0]))),
            )
        if isinstance(value, (tuple, list)):
            return (
                "sequence",
                _container_kind(value),
                _container_state_token(value, state),
                tuple(_semantic_token(item, state) for item in value),
            )
        if isinstance(value, (set, frozenset)):
            return (
                "set",
                _container_kind(value),
                _container_state_token(value, state),
                tuple(
                    sorted(
                        (_semantic_token(item, state) for item in value),
                        key=repr,
                    )
                ),
            )
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return (
                type(value).__module__,
                type(value).__qualname__,
                tuple(
                    (
                        field.name,
                        _semantic_token(getattr(value, field.name), state),
                    )
                    for field in dataclasses.fields(value)
                ),
            )
        if callable(value):
            return _callable_token(value, state)

        attributes = _object_state_token(value, state)
        if attributes is not None:
            return (
                type(value).__module__,
                type(value).__qualname__,
                attributes,
            )
        stable_repr = _ADDRESS_IN_REPR.sub("0x?", repr(value))
        return type(value).__module__, type(value).__qualname__, stable_repr
    finally:
        del state.active[value_id]


def semantic_token(value: Any) -> Any:
    """Return a deterministic, hashable description of a semantic value."""

    return _semantic_token(value, _TokenState())
