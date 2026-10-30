# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::std::tuple."""

from __future__ import annotations

import re

import cccl_common

import lldb

_TUPLE_PATTERN = re.compile(r"^cuda::std::tuple<.*>$")
_LEAF_INDEX_PATTERN = re.compile(r"__tuple_leaf<(\d+),")
InternalDict = dict[str, object]


def is_cuda_tuple(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = cccl_common.canonical_type_name(value_type)
    return _TUPLE_PATTERN.fullmatch(type_name) is not None


def _has_elements(value: lldb.SBValue) -> bool:
    # cuda::std::tuple<> can compile with no debug-visible __base_ member.
    base = (
        cccl_common.strip_reference_value(value)
        .GetNonSyntheticValue()
        .GetChildMemberWithName("__base_")
    )
    return base.IsValid() and bool(_leaf_bases(base.GetType()))


def tuple_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str:
    # --expand plus an empty summary renders non-empty tuples as "(type) {
    # [i] = ... }" instead of LLDB's inconsistent single-/multi-line default.
    # An empty tuple has no children to expand though, so it needs a visible
    # "{}" summary to stay distinguishable from an omitted child.
    return "" if _has_elements(value) else "{}"


def _leaf_bases(base_type: lldb.SBType) -> list[tuple[int, lldb.SBTypeMember]]:
    """Return each ``__tuple_leaf`` base class paired with its tuple index.

    Enumerated from the *type*, not a value's children: LLDB's child
    enumeration silently omits zero-size (empty base class optimization)
    subobjects, but the type system still reports them with a correct offset.
    """
    leaves = []
    for i in range(base_type.GetNumberOfDirectBaseClasses()):
        member = base_type.GetDirectBaseClassAtIndex(i)
        match = _LEAF_INDEX_PATTERN.search(member.GetName() or "")
        if match is None:
            continue
        leaves.append((int(match.group(1)), member))
    leaves.sort(key=lambda pair: pair[0])
    return leaves


def _base_field_type(tuple_type: lldb.SBType) -> lldb.SBType | None:
    """Return the ``__base_`` member's type, or ``None`` (e.g. cuda::std::tuple<>)."""
    for i in range(tuple_type.GetNumberOfFields()):
        field = tuple_type.GetFieldAtIndex(i)
        if field.GetName() == "__base_":
            return field.GetType()
    return None


def _tuple_type_name(tuple_type: lldb.SBType) -> str:
    """Reconstruct a tuple's display name, template arguments included.

    GCC's debug info drops a variadic class template's own template
    arguments, collapsing every specialization's name to
    ``cuda::std::tuple<>`` (Clang is unaffected). Each ``__tuple_leaf<N, T>``
    base class still carries its element type ``T`` though, so the argument
    list is rebuilt from those instead.
    """
    canonical = tuple_type.GetCanonicalType().GetUnqualifiedType()
    display_name = canonical.GetDisplayTypeName() or canonical.GetName() or ""
    prefix = display_name.split("<", 1)[0]
    base_type = _base_field_type(canonical)
    leaves = _leaf_bases(base_type) if base_type is not None else []
    element_names = [
        _element_type_name(member.GetType().GetTemplateArgumentType(1))
        for _, member in leaves
    ]
    return f"{prefix}<{', '.join(element_names)}>"


def _element_type_name(element_type: lldb.SBType) -> str:
    """Return a tuple element's display name, recursing into nested tuples.

    A nested tuple element loses its own arguments to the same GCC
    limitation. The unqualified name is only used to test for that case;
    the returned text keeps the element's own qualifiers (e.g. ``const
    char *``), unlike the outer tuple's own top-level qualifier, which is
    dropped everywhere for consistency (see ``is_cuda_tuple``).
    """
    recognized_name = (
        element_type.GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName() or ""
    )
    if _TUPLE_PATTERN.fullmatch(recognized_name) is not None:
        return _tuple_type_name(element_type)
    return element_type.GetDisplayTypeName() or element_type.GetName() or ""


def _leaf_element(
    base_value: lldb.SBValue, member: lldb.SBTypeMember, name: str
) -> lldb.SBValue:
    """Extract the element stored in one ``__tuple_leaf`` base subobject.

    ``__tuple_leaf`` either holds the element in a ``__value_`` member, or
    (empty base class optimization, for empty non-final element types)
    privately inherits it directly. An empty element shares its offset with
    whatever follows it, but LLDB's ``CreateChildAtOffset`` caches values by
    (parent, offset) and returns the first type ever requested there,
    mismatching later requests at that offset. Building the empty element
    from freestanding data instead avoids poisoning that cache; every other
    leaf keeps using ``CreateChildAtOffset`` normally.
    """
    leaf_type = member.GetType()
    if leaf_type.GetNumberOfFields() == 0:
        element_type = leaf_type.GetDirectBaseClassAtIndex(0).GetType()
        byte_size = element_type.GetByteSize() or 1
        data = lldb.SBData()
        error = lldb.SBError()
        data.SetData(error, bytes(byte_size), lldb.eByteOrderLittle, 8)
        return base_value.CreateValueFromData(name, data, element_type)

    leaf = base_value.CreateChildAtOffset(name, member.GetOffsetInBytes(), leaf_type)
    element = leaf.GetChildMemberWithName("__value_")
    # Reference elements print as a bare address unless dereferenced explicitly.
    if element.GetType().IsReferenceType():
        element = element.Dereference()
    return element


class TupleSyntheticProvider:
    """Expose cuda::std::tuple elements as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.leaves: list[tuple[int, lldb.SBTypeMember]] = []
        self.base: lldb.SBValue = lldb.SBValue()
        self.update()

    def update(self) -> bool:
        # cuda::std::tuple<> can compile with no debug-visible __base_ member.
        self.base = self.value.GetChildMemberWithName("__base_")
        self.leaves = _leaf_bases(self.base.GetType()) if self.base.IsValid() else []
        return True

    def num_children(self) -> int:
        return len(self.leaves)

    def has_children(self) -> bool:
        return len(self.leaves) != 0

    def get_type_name(self) -> str:
        # Also fixes an alloc_traits::value_type typedef that STL element
        # access can leave visible instead of the real tuple type.
        return _tuple_type_name(self.value.GetType())

    def get_child_index(self, name: str) -> int:
        if name.startswith("[") and name.endswith("]"):
            try:
                return int(name[1:-1])
            except ValueError:
                pass
        return -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index < 0 or index >= len(self.leaves):
            return None
        tuple_index, member = self.leaves[index]
        name = f"[{tuple_index}]"
        return _leaf_element(self.base, member, name).Clone(name)


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::std::tuple formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function {module}.tuple_summary "
        f"--recognizer-function {module}.is_cuda_tuple"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.TupleSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_tuple"
    )
