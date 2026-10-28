# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe CUB specialization values for deterministic cache keys."""

from __future__ import annotations

import dataclasses
import math
import re
import struct
from collections import defaultdict
from collections.abc import Callable, Mapping
from enum import Enum
from types import ModuleType
from typing import Any

_ADDRESS_IN_REPR = re.compile(r"(?<= at )0x[0-9a-fA-F]+")


def _defined_module_name(value: Any) -> str | None:
    """Use a module name only when the object supplies it as text."""

    module_name = getattr(value, "__module__", None)
    return module_name if isinstance(module_name, str) else None


@dataclasses.dataclass
class _TokenState:
    """Track recursion and reuse completed work within one token query.

    ``normalize`` lets a backend replace compiler objects with supported data.
    ``active`` maps object identities to their depth on the current path. This
    lets recursive helpers and self-referencing objects end in a cycle marker.
    ``completed`` keeps reusable tokens by traversal mode and object identity.
    Each entry retains its source object so Python cannot reuse that identity
    during this query.
    ``cycle_hits`` counts back-references. Store a result for reuse only if
    its own traversal found none, because cycle depths depend on the path.
    """

    normalize: Callable[[Any], Any] | None = None
    active: dict[int, int] = dataclasses.field(default_factory=dict)
    completed: dict[tuple[str, int], tuple[Any, Any]] = dataclasses.field(
        default_factory=dict
    )
    cycle_hits: int = 0


def _object_state_token(
    value: Any,
    state: _TokenState,
) -> tuple[Any, ...] | None:
    """Describe stored instance attributes, including inherited private slots.

    Python mangles private slot names using the defining class. Resolve those
    names before reading values, and skip slots that have not been assigned.
    Return ``None`` when these attributes expose no stored state.

    """

    tokenize = _semantic_token
    object_state: list[tuple[str, Any]] = []
    attributes = getattr(value, "__dict__", None)
    if attributes:
        object_state.append(("__dict__", tokenize(attributes, state)))

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
                else tokenize(slot_value, state)
            )
            object_state.append((storage_name, token))

    return tuple(object_state) if object_state else None


def _dataclass_fields_token(value, state, tokenize):
    """Describe the fields of one CUB specialization record."""
    return tuple(
        (field.name, tokenize(getattr(value, field.name), state))
        for field in dataclasses.fields(value)
    )


def _cycle_token(
    value: Any, back_reference_depth: int
) -> tuple[str, str, str, int]:
    """Identify a cycle by object kind and distance to its active ancestor."""

    return (
        "cycle",
        getattr(value, "__module__", type(value).__module__),
        getattr(value, "__qualname__", type(value).__qualname__),
        back_reference_depth,
    )


def _container_kind(value: Any) -> tuple[str | None, str]:
    """Keep a container subclass distinct from its built-in counterpart."""

    return _defined_module_name(type(value)), type(value).__qualname__


def _container_state_token(
    value: Any,
    state: _TokenState,
) -> tuple[tuple[str, Any], ...] | None:
    """Capture container state that equal elements alone cannot describe.

    This includes instance attributes and a ``defaultdict``'s factory, which
    determines the value supplied for a missing key.
    """

    tokenize = _semantic_token
    container_state = list(
        _object_state_token(
            value,
            state,
        )
        or ()
    )
    if isinstance(value, defaultdict):
        default_factory = value.default_factory
        default_factory_token = (
            ("self",)
            if default_factory is value
            else tokenize(default_factory, state)
        )
        container_state.append(("default_factory", default_factory_token))
    return tuple(container_state) if container_state else None


def _semantic_token(value: Any, state: _TokenState) -> Any:
    """Describe a value while sharing cycle detection and memoized results.

    Scalar cases preserve type-sensitive distinctions before recursion starts.
    Container tokens include their type, stored state, and elements. Sort
    mapping entries by encoded key and set elements by encoded value, using
    their ``repr`` strings as sort keys. Sequences retain element order.
    Type and module values contribute their qualified names. Python callback
    specialization is unsupported.
    """

    if state.normalize is not None:
        normalized = state.normalize(value)
        if normalized is not value:
            return _semantic_token(normalized, state)
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
                _dataclass_fields_token(value, state, _semantic_token),
            )
        elif isinstance(value, (staticmethod, classmethod)):
            raise TypeError(
                "callable values are not supported in specialization keys"
            )
        elif callable(value):
            raise TypeError(
                "Python callbacks are not supported in "
                "CUTLASS specialization keys"
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
        # A cycle token contains path-relative depth, so reuse only results
        # that did not encounter a back-reference while visiting this value.
        state.completed[memo_key] = (value, token)
    return token


def semantic_token(
    value: Any, *, normalize: Callable[[Any], Any] | None = None
) -> Any:
    """Describe a specialization value for comparison and hashing.

    Algorithms use this token to compare bound values and planning metadata.
    Each call
    starts a fresh traversal so previous queries cannot supply stale tokens.
    Floating-point tokens preserve zero signs and NaN bits.

    Parameters
    ----------
    value : object
        Scalar, container, dataclass, type, module, or state-bearing
        object used in a specialization. Other objects with no stored state
        use their type and ``repr``, with usual memory-address text removed.
    normalize : callable, optional
        Backend conversion applied to visited values, including referenced
        dependencies. Return the original object when no conversion is needed.
        A replacement is visited again and must eventually stop changing by
        identity. Describe compiler objects by their compilation inputs,
        without including compiler bookkeeping.

    Returns
    -------
    object
        A scalar or nested tuple description. Supported specialization values
        produce hashable tokens.

    Notes
    -----
    Callable inspection follows stored state and explicit bytecode references.
    It cannot establish the behavior of arbitrary computed lookups or external
    state. A backend must normalize values whose relevant contents the shared
    encoder cannot describe, such as compiler-specific callback wrappers.
    """

    return _semantic_token(value, _TokenState(normalize=normalize))
