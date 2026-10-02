# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ._symbols import semantic_token
from ._types import (
    TemplateParameter,
)


def _freeze_methods(
    methods: Iterable[Iterable[Any]],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(method) for method in methods)


def _freeze_semantic_value(value: Any, active: set[int]) -> Any:
    if not isinstance(value, (Mapping, tuple, list, set, frozenset, bytearray)):
        return value

    identity = id(value)
    if identity in active:
        raise ValueError(
            "specialization values must not contain container cycles"
        )
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            return MappingProxyType(
                {
                    _freeze_semantic_value(key, active): _freeze_semantic_value(
                        item, active
                    )
                    for key, item in value.items()
                }
            )
        if isinstance(value, (tuple, list)):
            return tuple(_freeze_semantic_value(item, active) for item in value)
        if isinstance(value, (set, frozenset)):
            return frozenset(
                _freeze_semantic_value(item, active) for item in value
            )
        return bytes(value)
    finally:
        active.remove(identity)


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            key: _freeze_semantic_value(value, set())
            for key, value in mapping.items()
        }
    )


@dataclass(frozen=True)
class TypeDefinition:
    """C++ source that must precede the generated primitive wrapper."""

    name: str
    code: str

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.name, semantic_token(self.code)


@dataclass(frozen=True, eq=False)
class Algorithm:
    """A specialized C++ cooperative primitive ready for materialization.

    ``specialization`` binds all template parameters and any auxiliary
    dependency values without performing backend lowering. Both the bindings
    and optional ``metadata`` are frozen when the algorithm is constructed.
    """

    struct_name: str
    method_name: str
    c_name: str
    includes: tuple[str, ...]
    template_parameters: tuple[TemplateParameter, ...]
    parameters: tuple[tuple[Any, ...], ...]
    specialization: Mapping[str, Any] = field(kw_only=True)
    metadata: Mapping[str, Any] = field(default_factory=dict, kw_only=True)
    type_definitions: tuple[TypeDefinition, ...] = ()
    fake_return: bool = False
    output_by_reference: bool = False
    _semantic_key: tuple[Any, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "includes", tuple(self.includes))
        object.__setattr__(
            self, "template_parameters", tuple(self.template_parameters)
        )
        object.__setattr__(self, "parameters", _freeze_methods(self.parameters))
        object.__setattr__(
            self, "type_definitions", tuple(self.type_definitions)
        )

        names = self.template_parameter_names
        if len(set(names)) != len(names):
            raise ValueError("template parameter names must be unique")
        missing = [name for name in names if name not in self.specialization]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Template argument(s) not provided: {joined}")

        object.__setattr__(
            self, "specialization", _freeze_mapping(self.specialization)
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(
            self,
            "_semantic_key",
            (
                (
                    self.struct_name,
                    self.method_name,
                    self.c_name,
                    self.includes,
                    semantic_token(self.template_parameters),
                    semantic_token(self.parameters),
                    semantic_token(self.type_definitions),
                    self.fake_return,
                    self.output_by_reference,
                ),
                semantic_token(self.template_arguments),
                semantic_token(self.metadata),
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Algorithm):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def template_parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.template_parameters)

    @property
    def template_arguments(self) -> Mapping[str, Any]:
        return self.specialization

    @property
    def ordered_template_arguments(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (name, self.template_arguments[name])
            for name in self.template_parameter_names
        )

    @property
    def ordered_specialization_arguments(self) -> tuple[tuple[str, Any], ...]:
        """Template arguments followed by deterministically ordered
        auxiliaries.
        """

        template_names = set(self.template_parameter_names)
        auxiliary = sorted(
            (
                (name, value)
                for name, value in self.template_arguments.items()
                if name not in template_names
            ),
            key=lambda item: item[0],
        )
        return (*self.ordered_template_arguments, *auxiliary)

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """Stable, hashable identity for provider/cache de-duplication."""

        return self._semantic_key
