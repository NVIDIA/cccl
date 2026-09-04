# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral cooperative algorithm descriptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ._symbols import semantic_token
from ._types import ParameterClassification, TemplateParameter, classify_parameter


def _freeze_methods(methods: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(method) for method in methods)


def _freeze_semantic_value(value: Any, active: set[int]) -> Any:
    if not isinstance(value, (Mapping, tuple, list, set, frozenset, bytearray)):
        return value

    identity = id(value)
    if identity in active:
        raise ValueError("specialization values must not contain container cycles")
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
            return frozenset(_freeze_semantic_value(item, active) for item in value)
        return bytes(value)
    finally:
        active.remove(identity)


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _freeze_semantic_value(value, set()) for key, value in mapping.items()}
    )


@dataclass(frozen=True)
class TypeDefinition:
    """C++ source that must precede the generated primitive wrapper."""

    name: str
    code: str

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.name, semantic_token(self.code)


@dataclass(frozen=True)
class Algorithm:
    """Unspecialized C++ cooperative primitive semantics."""

    struct_name: str
    method_name: str
    c_name: str
    includes: tuple[str, ...]
    template_parameters: tuple[TemplateParameter, ...]
    parameters: tuple[tuple[Any, ...], ...]
    type_definitions: tuple[TypeDefinition, ...] = ()
    fake_return: bool = False
    output_by_reference: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "includes", tuple(self.includes))
        object.__setattr__(self, "template_parameters", tuple(self.template_parameters))
        object.__setattr__(self, "parameters", _freeze_methods(self.parameters))
        object.__setattr__(self, "type_definitions", tuple(self.type_definitions))

        names = self.template_parameter_names
        if len(set(names)) != len(names):
            raise ValueError("template parameter names must be unique")

    @property
    def template_parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.template_parameters)

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.struct_name,
            self.method_name,
            self.c_name,
            self.includes,
            semantic_token(self.template_parameters),
            semantic_token(self.parameters),
            semantic_token(self.type_definitions),
            self.fake_return,
            self.output_by_reference,
        )

    def specialize(
        self,
        template_arguments: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "AlgorithmSpec":
        """Bind template/dependency values without performing backend lowering."""

        missing = [
            name
            for name in self.template_parameter_names
            if name not in template_arguments
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Template argument(s) not provided: {joined}")
        return AlgorithmSpec(
            algorithm=self,
            template_arguments=template_arguments,
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True, eq=False)
class AlgorithmSpec:
    """A specialized algorithm ready for backend materialization."""

    algorithm: Algorithm
    template_arguments: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _semantic_key: tuple[Any, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "template_arguments",
            _freeze_mapping(self.template_arguments),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(
            self,
            "_semantic_key",
            (
                self.algorithm.semantic_key,
                semantic_token(self.template_arguments),
                semantic_token(self.metadata),
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlgorithmSpec):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def struct_name(self) -> str:
        return self.algorithm.struct_name

    @property
    def method_name(self) -> str:
        return self.algorithm.method_name

    @property
    def c_name(self) -> str:
        return self.algorithm.c_name

    @property
    def includes(self) -> tuple[str, ...]:
        return self.algorithm.includes

    @property
    def template_parameters(self) -> tuple[TemplateParameter, ...]:
        return self.algorithm.template_parameters

    @property
    def template_parameter_names(self) -> tuple[str, ...]:
        return self.algorithm.template_parameter_names

    @property
    def parameters(self) -> tuple[tuple[Any, ...], ...]:
        return self.algorithm.parameters

    @property
    def type_definitions(self) -> tuple[TypeDefinition, ...]:
        return self.algorithm.type_definitions

    @property
    def fake_return(self) -> bool:
        return self.algorithm.fake_return

    @property
    def output_by_reference(self) -> bool:
        return self.algorithm.output_by_reference

    @property
    def ordered_template_arguments(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (name, self.template_arguments[name])
            for name in self.template_parameter_names
        )

    @property
    def ordered_specialization_arguments(self) -> tuple[tuple[str, Any], ...]:
        """Template arguments followed by deterministically ordered auxiliaries."""

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
    def symbol_mangling_inputs(self) -> tuple[Any, ...]:
        """Raw semantic inputs a backend should use when naming a shim."""

        return self.c_name, self.method_name, self.semantic_key

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """Stable, hashable identity for provider/cache de-duplication."""

        return self._semantic_key

    def classify_method(
        self,
        method_index: int = 0,
    ) -> tuple[ParameterClassification, ...]:
        return tuple(
            classify_parameter(parameter) for parameter in self.parameters[method_index]
        )
