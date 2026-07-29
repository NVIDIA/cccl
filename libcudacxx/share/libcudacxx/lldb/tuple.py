# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::std::tuple."""

from __future__ import annotations

import re

import lldb

_TUPLE_PATTERN = re.compile(r"^cuda::std::tuple<.*>$")
_LEAF_INDEX_PATTERN = re.compile(r"__tuple_leaf<(\d+),")
InternalDict = dict[str, object]


def is_cuda_tuple(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = (
        value_type.GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName() or ""
    )
    return _TUPLE_PATTERN.fullmatch(type_name) is not None


def _leaf_bases(base_type: lldb.SBType) -> list[tuple[int, lldb.SBTypeMember]]:
    """Return each ``__tuple_leaf`` base class paired with its tuple index.

    Direct base classes are enumerated from the *type* rather than from a
    value's children: LLDB's child enumeration silently omits zero-size base
    subobjects (the empty base class optimization case below), while the type
    system still reports them with a correct offset.
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


def _leaf_element(
    base_value: lldb.SBValue, member: lldb.SBTypeMember, name: str
) -> lldb.SBValue:
    """Extract the element stored in one ``__tuple_leaf`` base subobject.

    ``__tuple_leaf`` either holds the element in a ``__value_`` data member, or
    (for empty, non-final element types) applies the empty base class
    optimization and privately inherits from the element type directly.

    Two (or more) leaves can share the same offset when a leading element is
    empty (the empty base class optimization): the element occupies no
    storage of its own and overlaps whatever follows it. LLDB's
    ``CreateChildAtOffset`` caches values by (parent, offset) and returns the
    first type ever requested at a given offset, silently mismatching later
    requests at that same offset. Building the empty element from
    freestanding, address-independent data instead of reading through the
    parent avoids poisoning that cache; every other (non-empty, uniquely
    offset) leaf keeps using ``CreateChildAtOffset`` normally.
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
        self.value = value.GetNonSyntheticValue()
        self.leaves: list[tuple[int, lldb.SBTypeMember]] = []
        self.base: lldb.SBValue = lldb.SBValue()
        self.update()

    def update(self) -> bool:
        self.base = self.value.GetChildMemberWithName("__base_")
        # A fully empty tuple (e.g. cuda::std::tuple<>) can compile to a type
        # with no debug-visible __base_ member at all.
        self.leaves = _leaf_bases(self.base.GetType()) if self.base.IsValid() else []
        return True

    def num_children(self) -> int:
        return len(self.leaves)

    def has_children(self) -> bool:
        return len(self.leaves) != 0

    def get_type_name(self) -> str:
        # STL element access can preserve an alloc_traits::value_type typedef.
        # Report the canonical display name so LLDB shows cuda::std::tuple
        # instead.
        return (
            self.value.GetType()
            .GetCanonicalType()
            .GetUnqualifiedType()
            .GetDisplayTypeName()
            or ""
        )

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
        f"type synthetic add --category {category} --python-class {module}.TupleSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_tuple"
    )
