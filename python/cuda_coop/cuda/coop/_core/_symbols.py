# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build cache keys from cooperative specialization values."""

from __future__ import annotations

import dataclasses
import math
import re
import struct
from collections import defaultdict
from collections.abc import Mapping
from enum import Enum
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from typing import Any

_ADDRESS_IN_REPR = re.compile(r"(?<= at )0x[0-9a-fA-F]+")


def _defined_module_name(value: Any) -> str | None:
    module_name = getattr(value, "__module__", None)
    return module_name if isinstance(module_name, str) else None


@dataclasses.dataclass
class _TokenState:
    active: dict[int, int] = dataclasses.field(default_factory=dict)
    completed: dict[tuple[str, int], tuple[Any, Any]] = dataclasses.field(
        default_factory=dict
    )
    cycle_hits: int = 0


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
                ("self",)
                if slot_value is value
                else _semantic_token(slot_value, state)
            )
            object_state.append((storage_name, token))

    return tuple(object_state) if object_state else None


def _cycle_token(
    value: Any, back_reference_depth: int
) -> tuple[str, str, str, int]:
    return (
        "cycle",
        getattr(value, "__module__", type(value).__module__),
        getattr(value, "__qualname__", type(value).__qualname__),
        back_reference_depth,
    )


def _container_kind(value: Any) -> tuple[str | None, str]:
    return _defined_module_name(type(value)), type(value).__qualname__


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
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, float):
        if math.isnan(value):
            return "float", "nan", struct.pack(">d", value).hex()
        return "float", value.hex()
    if value is None or isinstance(value, (int, str, bytes)):
        return value
    if isinstance(value, ModuleType):
        return "module", value.__name__
    if isinstance(value, type):
        return "type", _defined_module_name(value), value.__qualname__

    value_id = id(value)
    if value_id in state.active:
        state.cycle_hits += 1
        return _cycle_token(value, len(state.active) - state.active[value_id])

    memo_key = ("value", value_id)
    completed = state.completed.get(memo_key)
    if completed is not None:
        return completed[1]

    state.active[value_id] = len(state.active)
    cycle_hits_before = state.cycle_hits
    try:
        if isinstance(value, Mapping):
            items_snapshot = tuple(value.items())
            items = [
                (_semantic_token(key, state), _semantic_token(item, state))
                for key, item in items_snapshot
            ]
            token = (
                "mapping",
                _container_kind(value),
                _container_state_token(value, state),
                tuple(sorted(items, key=lambda item: repr(item[0]))),
            )
        elif isinstance(value, (tuple, list)):
            token = (
                "sequence",
                _container_kind(value),
                _container_state_token(value, state),
                tuple(_semantic_token(item, state) for item in value),
            )
        elif isinstance(value, (set, frozenset)):
            token = (
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
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            token = (
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
        elif isinstance(value, (BuiltinFunctionType, FunctionType, MethodType)):
            raise TypeError(
                "callable values are not supported in specialization keys"
            )
        else:
            attributes = _object_state_token(value, state)
            if attributes is not None:
                token = (
                    type(value).__module__,
                    type(value).__qualname__,
                    attributes,
                )
            else:
                stable_repr = _ADDRESS_IN_REPR.sub("0x?", repr(value))
                token = (
                    type(value).__module__,
                    type(value).__qualname__,
                    stable_repr,
                )
    finally:
        del state.active[value_id]

    if state.cycle_hits == cycle_hits_before:
        state.completed[memo_key] = (value, token)
    return token


def semantic_token(value: Any) -> Any:
    """Return a deterministic, hashable description of a semantic value."""

    return _semantic_token(value, _TokenState())
