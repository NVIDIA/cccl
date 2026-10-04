# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Find where ``ThreadData`` and ``TempStorage`` values came from in the IR.

These objects describe per-thread payloads and shared scratch for the
compiler; they do not become ordinary Python objects in a GPU kernel. To
replace them with arrays and storage pointers, the compiler must recover their
constructors and options even when the kernel assigns aliases, casts values,
or joins branches. A "reaching definition" is an assignment that may supply a
variable's value; "provenance" here means following those assignments back to
their sources.

The helpers trace descriptor sources, bind ``TempStorage`` constructor
options, and collect the element types written through payload aliases. Group
planning uses these facts to select an operation implementation; call
rewriting uses them to materialize payloads and scratch storage. Each caller
supplies its own definition and constant lookups because the available IR
facts differ between those phases. The shared traversal keeps their treatment
of aliases and control-flow joins consistent without evaluating arbitrary
kernel code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numba_cuda_mlir.numba_cuda.core import ir

    from .._temp_storage import TempStorage
else:
    from numba_cuda_mlir.numbair_transforms import ir


def descriptor_definitions(
    value: object,
    definitions: Callable[[ir.Var], Iterable[object]],
    *,
    seen: set[str] | None = None,
) -> Iterator[tuple[str | None, object]]:
    """Trace a possible descriptor to the assignments that supply its value.

    Follow aliases, casts, iterator unpacking, and phi inputs using the
    caller's definition lookup, so the same traversal works before and after
    single static assignment (SSA) construction. A cycle contributes no leaf.
    Concrete non-descriptors, including ``None``, remain leaves: callers need
    to reject paths that mix such values with descriptors rather than
    accepting one valid constructor. Sibling paths are independent, so a
    shared constructor may be yielded more than once.

    Parameters
    ----------
    value : object
        IR value to trace. A non-variable is yielded unchanged with no owner.
    definitions : callable
        Lookup accepting an IR variable and returning all its reaching
        definitions, including multiple definitions before SSA construction.
    seen : set of str, optional
        Variable names on the current recursion path. This traversal copies
        the set before extending it and does not mutate the supplied set.

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
    call: ir.Expr,
    constant: Callable[..., Any],
    *,
    syntax_error: type[Exception] = TypeError,
) -> TempStorage:
    """Build a storage descriptor from an IR constructor call.

    Share argument binding and descriptor validation between group planning
    and provider rewriting while letting each phase supply its own constant
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


def payload_write_dtypes(
    func_ir: ir.FunctionIR,
    payload: object,
    dtype: Callable[[ir.Var], object],
) -> Iterator[object]:
    """Yield known element dtypes written through a payload's aliases.

    Build an alias set to a fixed point across the entire function, following
    assignments, casts, iterator unpacking, and phi inputs in both directions.
    This lets inference start at either a constructor or a later alias. The
    scan is flow-insensitive: it collects writes throughout the function,
    including all connected phi inputs, without checking path feasibility or
    write order. The caller decides whether the collected dtypes agree.

    Parameters
    ----------
    func_ir : ir.FunctionIR
        Function with alias assignments and element writes; not modified.
    payload : object
        Variable whose aliases are searched; non-variables yield nothing.
    dtype : callable
        Lookup accepting the variable assigned to an element and returning its
        dtype, or ``None`` when unknown.

    Yields
    ------
    object
        Known dtype for each ``SetItem`` or ``StaticSetItem`` write through
        the alias set, in the representation supplied by ``dtype``. This
        helper does not normalize the callback's results. Unknown dtypes are
        skipped; duplicates are retained. An empty result does not establish
        that the payload has no writes.
    """

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
