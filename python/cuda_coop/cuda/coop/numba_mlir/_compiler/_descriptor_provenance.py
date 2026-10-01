# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reaching definitions for opaque descriptors before and after SSA."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numba_cuda_mlir.numba_cuda.core import ir
else:
    from numba_cuda_mlir.numbair_transforms import ir


def descriptor_definitions(value, definitions, *, seen=None):
    """Yield the leaves that can define an opaque descriptor.

    Follow aliases, casts, iterator unpacking, and phi inputs using the caller's
    definition lookup, so the same traversal works before and after SSA
    construction. A cycle contributes no leaf. A concrete non-descriptor,
    including ``None``, remains a leaf: callers need to reject paths that mix
    such values with descriptors rather than silently accepting one valid
    constructor. Sibling paths are independent, so a shared constructor may be
    yielded more than once.

    Parameters
    ----------
    value : ir.Var or object
        IR value to trace. A non-variable is yielded unchanged with no owner.
    definitions : callable
        Lookup accepting an IR variable and returning all its reaching
        definitions, including multiple definitions before SSA construction.
    seen : set of str, optional
        Variable names on the current recursion path. This traversal copies the
        set before extending it and does not mutate the supplied set.

    Yields
    ------
    owner : str or None
        Name of the variable whose definition is the leaf, not necessarily the
        original alias. ``None`` denotes a non-variable input.
    leaf : object
        Definition reached after following the supported forwarding forms. No
        descriptor recognition or constructor validation is performed.
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


def temp_storage_constructor(
    call, constant, *, syntax_error: type[Exception] = TypeError
):
    """Build a storage descriptor from an IR constructor call.

    Share argument binding and descriptor validation between group planning and
    provider rewriting while letting each phase supply its own constant
    resolver. This creates the host-side descriptor only; it does not allocate
    scratch or modify the call. Constructor defaults and value validation come
    from ``TempStorage`` itself.

    Parameters
    ----------
    call : ir.Expr
        Call already recognized as a ``TempStorage`` constructor.
    constant : callable
        Resolve an argument as ``constant(value, name=parameter_name)``. The
        callback controls constant specialization and its diagnostics.
    syntax_error : type of Exception, optional
        Exception class for unsupported call syntax, duplicate arguments, or
        unknown keywords. Defaults to ``TypeError``; exceptions from the
        resolver and descriptor constructor propagate unchanged.

    Returns
    -------
    TempStorage
        Descriptor with resolved, validated constructor arguments.

    Raises
    ------
    TypeError
        Invalid call syntax when ``syntax_error`` has its default value, or an
        invalid descriptor option type.
    ValueError
        A descriptor option has an invalid value.
    """

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
