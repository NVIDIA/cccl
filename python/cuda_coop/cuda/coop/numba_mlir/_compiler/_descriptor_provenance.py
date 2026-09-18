# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reaching definitions for opaque descriptors before and after SSA."""

from ._numba_mlir_compat import _get_numba_mlir_compat

ir = _get_numba_mlir_compat().numba_ir


def descriptor_definitions(value, definitions, *, seen=None):
    """Yield (owner, leaf) pairs through aliases, casts, and conditional joins.

    A backedge has no new definition and contributes no leaf. A concrete
    non-descriptor, including a constant None, remains a leaf so callers can
    distinguish it from an unresolved cycle. Visit sibling paths separately:
    a constructor reached twice is still a descriptor on both paths.
    """

    if not isinstance(value, ir.Var):
        yield None, value
        return
    if seen is None:
        seen = set()
    if value.name in seen:
        return
    seen = {*seen, value.name}
    for definition in definitions(value):
        if isinstance(definition, ir.Var):
            yield from descriptor_definitions(definition, definitions, seen=seen)
        elif isinstance(definition, ir.Expr) and definition.op in {
            "cast",
            "exhaust_iter",
        }:
            yield from descriptor_definitions(definition.value, definitions, seen=seen)
        elif isinstance(definition, ir.Expr) and definition.op == "phi":
            for incoming in getattr(definition, "incoming_values", ()):
                yield from descriptor_definitions(incoming, definitions, seen=seen)
        else:
            yield value.name, definition
