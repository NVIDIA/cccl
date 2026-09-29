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
            yield from descriptor_definitions(
                definition, definitions, seen=seen
            )
        elif isinstance(definition, ir.Expr) and definition.op in {
            "cast",
            "exhaust_iter",
        }:
            yield from descriptor_definitions(
                definition.value, definitions, seen=seen
            )
        elif isinstance(definition, ir.Expr) and definition.op == "phi":
            for incoming in getattr(definition, "incoming_values", ()):
                yield from descriptor_definitions(
                    incoming, definitions, seen=seen
                )
        else:
            yield value.name, definition


def temp_storage_constructor(call, constant, *, syntax_error=TypeError):
    """Parse a descriptor using the current phase's constant resolver."""

    from .._temp_storage import TempStorage

    if (
        getattr(call, "vararg", None) is not None
        or getattr(call, "varkwarg", None) is not None
    ):
        raise syntax_error(
            "TempStorage does not accept *args or **kwargs; pass "
            "size_in_bytes, alignment, auto_sync, and sharing explicitly."
        )
    if len(call.args) > 1:
        raise syntax_error(
            "TempStorage accepts only size_in_bytes positionally; "
            "alignment, auto_sync, and sharing are keyword-only."
        )
    refs = dict(zip(("size_in_bytes",), call.args))
    for name, value in call.kws:
        if name in refs:
            raise syntax_error(
                f"TempStorage got multiple values for argument {name!r}"
            )
        refs[name] = value
    unexpected = sorted(
        refs.keys() - {"size_in_bytes", "alignment", "auto_sync", "sharing"}
    )
    if unexpected:
        raise syntax_error(
            f"TempStorage got unexpected keyword(s): {', '.join(unexpected)}"
        )
    return TempStorage(
        **{name: constant(value, name=name) for name, value in refs.items()}
    )


def payload_write_dtypes(func_ir, payload, dtype):
    """Yield known dtypes written through a payload's reaching aliases."""

    if not isinstance(payload, ir.Var):
        return
    aliases = {payload.name}
    changed = True
    while changed:
        changed = False
        for block in func_ir.blocks.values():
            for statement in block.body:
                if not isinstance(statement, ir.Assign):
                    continue
                definition = statement.value
                sources = ()
                if isinstance(definition, ir.Var):
                    sources = (definition,)
                elif isinstance(definition, ir.Expr):
                    if definition.op in {"cast", "exhaust_iter"}:
                        sources = (definition.value,)
                    elif definition.op == "phi":
                        sources = getattr(definition, "incoming_values", ())
                source_names = {
                    source.name
                    for source in sources
                    if isinstance(source, ir.Var)
                }
                if statement.target.name in aliases or source_names & aliases:
                    additions = {statement.target.name, *source_names} - aliases
                    if additions:
                        aliases.update(additions)
                        changed = True

    for block in func_ir.blocks.values():
        for statement in block.body:
            if not isinstance(statement, (ir.SetItem, ir.StaticSetItem)):
                continue
            if (
                isinstance(statement.target, ir.Var)
                and statement.target.name in aliases
                and isinstance(statement.value, ir.Var)
            ):
                value_dtype = dtype(statement.value)
                if value_dtype is not None:
                    yield value_dtype
