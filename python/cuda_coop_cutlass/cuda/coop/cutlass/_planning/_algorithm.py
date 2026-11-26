# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe a concrete primitive before a backend generates its callable code.

Factories bind template arguments and describe the C++ method's parameters in
an ``Algorithm``. Backends translate that description into their own compiler
types and generated wrappers. The core record lets factories share this work
without importing a compiler. Its identity includes the bound values so cache
lookups can distinguish different specializations of the same primitive.
"""

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
    """Copy method sequences into tuples and retain their descriptors."""

    return tuple(tuple(method) for method in methods)


def _freeze_semantic_value(value: Any, active: set[int]) -> Any:
    """Copy supported containers so later caller edits cannot change them.

    Mappings become read-only mappings, lists and tuples become tuples, sets
    become frozen sets, and byte arrays become bytes. Other objects stay as
    supplied. ``active`` holds container identities on the current recursion
    path. A cycle raises ``ValueError``; repeated references are allowed.
    """

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
    """Snapshot named values for an algorithm's cached identity."""

    return MappingProxyType(
        {
            key: _freeze_semantic_value(value, set())
            for key, value in mapping.items()
        }
    )


@dataclass(frozen=True)
class TypeDefinition:
    """Supply a C++ declaration needed by the generated primitive wrapper.

    A factory can use this for a helper type referenced by its parameters or
    template arguments. The backend emits ``code`` before the wrapper.

    Attributes
    ----------
    name : str
        Name that identifies this definition in the algorithm description.
    code : str
        Complete C++ source for the declaration, including any terminator.
    """

    name: str
    code: str

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.name, semantic_token(self.code)


@dataclass(frozen=True, eq=False)
class Algorithm:
    """Describe one C++ primitive with its template arguments already bound.

    Construction checks that every declared template name has a binding.
    A backend then resolves parameter dependencies and builds the callable
    wrapper. This record contains the information for that step; constructing
    it does not compile or link code.

    Attributes
    ----------
    struct_name : str
        C++ primitive class name, such as ``BlockLoad``.
    method_name : str
        Method to call on that class, such as ``Load``.
    c_name : str
        Stem the backend uses to name generated wrappers.
    includes : tuple of str
        Header paths needed to compile the primitive and its parameters.
    template_parameters : tuple of TemplateParameter
        Template declarations in C++ argument order.
    parameters : tuple of tuple
        Candidate method signatures. Each inner tuple contains parameter
        descriptors in call order, including any temporary-storage marker.
    template_arguments : Mapping
        Values for the declared template parameters and any extra named
        dependencies used by parameter descriptors.
    metadata : Mapping
        Additional planning facts. These also contribute to record identity.
    type_definitions : tuple of TypeDefinition
        Helper declarations the backend emits before the wrapper.
    fake_return : bool
        If true, pass every output descriptor to the C++ method as an ordinary
        argument. The wrapper ignores the method's return value.
    output_by_reference : bool
        Used when ``fake_return`` is false. The signature can then have at
        most one output, which receives the result. ``True`` passes that output
        to the method as an argument. ``False`` leaves it out of the method's
        arguments and assigns the method's return value to it.

    Notes
    -----
    Construction copies supported containers in ``template_arguments`` and
    ``metadata`` into immutable forms, then computes the equality and hash
    key. It retains other leaf objects as supplied. Parameter descriptors and
    those leaf objects must remain suitable for the backend that reads them.

    Extra dependency names need not be C++ template parameters. For example,
    a parameter's array extent can use a named value that the class template
    itself does not accept.

    Return-handling flags describe the wrapper behavior for the adapter to
    implement. The core stores these flags and includes them in identity;
    wrapper generation belongs to the backend.
    """

    struct_name: str
    method_name: str
    c_name: str
    includes: tuple[str, ...]
    template_parameters: tuple[TemplateParameter, ...]
    parameters: tuple[tuple[Any, ...], ...]
    template_arguments: Mapping[str, Any] = field(kw_only=True)
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
        missing = [
            name for name in names if name not in self.template_arguments
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Template argument(s) not provided: {joined}")

        object.__setattr__(
            self, "template_arguments", _freeze_mapping(self.template_arguments)
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
    def ordered_template_arguments(self) -> tuple[tuple[str, Any], ...]:
        """Return template bindings in C++ order."""

        return tuple(
            (name, self.template_arguments[name])
            for name in self.template_parameter_names
        )

    @property
    def ordered_specialization_arguments(self) -> tuple[tuple[str, Any], ...]:
        """List template bindings first, then extra dependencies by name.

        Backends use this order to resolve a signature without depending on
        the caller's mapping insertion order.
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
        """Return the cached identity of the description and bound values.

        Equality and hashing use this key so equivalent records can share a
        provider or cache entry. Construction computes it once.
        """

        return self._semantic_key
