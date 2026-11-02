# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type and value helpers shared by the CCCL GDB pretty printers."""

from __future__ import annotations

import re

import gdb

_ABI_NAMESPACE_PATTERN = re.compile(r"::__(?:\d+|version_bump_ver\d+_)(?=::)")
_REFERENCE_CODES = (gdb.TYPE_CODE_REF, gdb.TYPE_CODE_RVALUE_REF)
# GDB writes non-type template arguments as C literals, so extents<int, 3ul, 4ul>
# breaks anything that parses the name.
_INTEGER_SUFFIX_PATTERN = re.compile(r"\b(\d+)[uUlL]+\b")


def public_type_name(value_type: gdb.Type) -> str:
    """Return a type name without CUDA ABI inline namespaces or integer suffixes."""
    name = _ABI_NAMESPACE_PATTERN.sub("", str(value_type))
    return _INTEGER_SUFFIX_PATTERN.sub(r"\1", name)


def template_name(value_type: gdb.Type | str) -> str:
    """Return the part of a type name before the template argument list."""
    name = value_type if isinstance(value_type, str) else str(value_type)
    return name.split("<", 1)[0]


def strip_reference(value_type: gdb.Type) -> gdb.Type:
    """Return the referenced type, or the type itself if it is not a reference.

    target() gives the referenced type, but on a non-reference it either
    resolves a typedef or raises, so test the type code first.
    """
    if value_type.code in _REFERENCE_CODES:
        return value_type.target()
    return value_type


def strip_reference_value(value: gdb.Value) -> gdb.Value:
    """Return the referenced value, or the value itself if it is not a reference."""
    if value.type.code in _REFERENCE_CODES:
        return value.referenced_value()
    return value


def canonical_type(value_type: gdb.Type) -> gdb.Type:
    """Return the type behind any reference, typedef, or cv-qualifier."""
    return strip_reference(value_type).strip_typedefs().unqualified()


def canonical_type_name(value_type: gdb.Type) -> str:
    """Return the public name of the type behind any reference or typedef."""
    return public_type_name(canonical_type(value_type))
