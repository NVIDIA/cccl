# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type and value helpers shared by the CCCL LLDB pretty printers."""

from __future__ import annotations

import re

import lldb

_ABI_NAMESPACE_PATTERN = re.compile(r"::__(?:\d+|version_bump_ver\d+_)(?=::)")


def strip_reference(value_type: lldb.SBType) -> lldb.SBType:
    """Return the type behind any reference, typedef, or cv-qualifier.

    GetDereferencedType() is a no-op on a non-reference, so it needs no guard.
    """
    return value_type.GetCanonicalType().GetDereferencedType().GetUnqualifiedType()


def strip_reference_value(value: lldb.SBValue) -> lldb.SBValue:
    """Return the referenced value, or the value itself if it is not a reference."""
    if value.GetType().IsReferenceType():
        return value.Dereference()
    return value


def canonical_type_name(value_type: lldb.SBType) -> str:
    """Return the display name of the type behind any reference or typedef."""
    return strip_reference(value_type).GetDisplayTypeName() or ""


def public_type_name(value_type: lldb.SBType) -> str:
    """Return the complete type name without CUDA ABI inline namespaces.

    GetName() keeps default template arguments that GetDisplayTypeName() hides.
    """
    value_type = strip_reference(value_type)
    type_name = value_type.GetName() or value_type.GetDisplayTypeName() or ""
    return _ABI_NAMESPACE_PATTERN.sub("", type_name)
