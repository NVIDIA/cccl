# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Put the resolved storage into the IR and tell lowering what to call.

Storage materialization is the last stage of ``CoopCallResolution``, and the
only one that changes the IR. It:

* selects each site's provider method and pairs its operands with where
  lowering finds their values, as a ``CallEntry``;
* prepends the scratch backing, a ``cuda.shared.array``, and the region's
  slice of it to the kernel's entry block, with their types;
* passes the region to every call that uses scratch, as the
  ``SCRATCH_KEYWORD`` keyword argument.

It takes no decisions and cannot fail on a user's kernel: every call entry
is built before the IR is touched.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, cast

from . import _bridges as nb
from ._link import Bundle
from ._lower import SCRATCH_KEYWORD
from ._plan import KernelPlan
from ._sites import Site
from ._specialize import Specialization
from ._storage import StorageLayout


@dataclass(frozen=True)
class CallEntry:
    """What lowering emits for one call site.

    ``operands`` follows ``abi_arguments``, one entry per provider operand:

    ``("scratch",)``
        the scratch region's pointer;
    ``("pointer", source)``
        the data pointer of a positional argument (integer ``source``) or
        keyword argument (string ``source``), including callback state;
    ``("box", source, dtype, value)``
        a one-item array of ``dtype`` holding the scalar argument ``source``,
        or the static ``value`` when it is not ``None``;
    ``("value", source)``
        a positional or keyword scalar argument, converted to the ABI type.
    ``("allocation", index)``
        the pointer to a fresh local array described by ``allocations``.
    ``("length", source)``
        the first dimension of an array, converted to the ABI scalar type.
    ``("constant", value)``
        a checked scalar constant represented in the provider ABI dtype.

    ``return_type`` is the provider's scalar ABI result, or ``void``.
    ``allocations`` and ``result`` separately describe fresh local arrays
    and the public result assembly; see ``_results``.
    """

    symbol: str
    abi_arguments: tuple
    operands: tuple
    return_type: Any
    barrier_after: bool
    link_items: tuple
    allocations: tuple = ()
    result: tuple | None = None


@dataclass
class CoopPlan:
    """``state.metadata["coop_plan"]``: the plan and its lowering data.

    ``calls`` is keyed by the name of each call's target variable.
    """

    kernel: KernelPlan
    calls: dict[str, CallEntry] = field(default_factory=dict)


# Methods and operands ---------------------------------------------------------


def _ignored_parameter(parameter: Any) -> bool:
    from .. import _types as coop_types

    if isinstance(
        parameter,
        (
            coop_types.CxxFunction,
            coop_types.StatelessOperator,
        ),
    ):
        return True
    return (
        isinstance(parameter, coop_types.PointerOffset)
        and parameter.static_value is not None
    )


def _inputs(method: Any) -> list:
    return [p for p in method if not _ignored_parameter(p) and not p.is_output]


def _pointer_parameter(parameter: Any) -> bool:
    from .. import _types as coop_types

    return isinstance(
        parameter, (coop_types.Pointer, coop_types.StatefulOperator)
    )


def _select_method(algorithm: Any, operands: tuple) -> Any:
    for method in algorithm.parameters:
        inputs = _inputs(method)
        if len(inputs) == len(operands) and all(
            _pointer_parameter(parameter)
            == (operand[0] in {"scratch", "pointer", "box", "allocation"})
            for parameter, operand in zip(inputs, operands, strict=True)
        ):
            return method
    raise AssertionError(
        f"{algorithm.c_name} has no method for operands {operands!r}"
    )


def _request_operands(site: Site, scratch: bool) -> tuple:
    """Bind call operands without assuming every operation is Load/Store.

    New requests provide their runtime inputs in provider order, excluding
    scratch and static functors. Load/Store retains its existing binding.
    """

    leading = (("scratch",),) if scratch else ()
    explicit = getattr(site.request, "provider_operands", None)
    if explicit is not None:
        return leading + tuple(explicit)
    from ._load_store import EFFECTIVE_OFFSET_KEYWORD

    warp = not site.request.group.is_block
    payload = (
        ("pointer", 2)
        if site.scalar is None
        else ("box", 2, site.request.dtype, site.scalar.value)
    )
    runtime = tuple(
        ("value", name)
        for name, binding in site.request.controls
        if binding.kind.value == "runtime" and not (warp and name == "offset")
    )
    if warp:
        runtime += (("value", EFFECTIVE_OFFSET_KEYWORD),)
    return leading + (("pointer", 1), payload) + runtime


def _return_type(method: Any) -> Any:
    outputs = [
        p.dtype() for p in method if p.is_output and not _ignored_parameter(p)
    ]
    if len(outputs) > 1:
        raise ValueError("Multiple output parameters not supported")
    result = outputs[0] if outputs else nb.types.void
    if isinstance(result, (nb.types.Array, nb.types.BaseTuple)):
        raise NotImplementedError(
            "cooperative array and pair results require "
            "a result allocation plan"
        )
    return result


def _call_entry(
    site: Site, algorithm: Any, scratch: bool, barrier: bool, bundle: Bundle
) -> CallEntry:
    operands = _request_operands(site, scratch)
    method = _select_method(algorithm, operands)
    abi_arguments = tuple(
        (
            nb.types.CPointer(nb.types.none)
            if _pointer_parameter(parameter)
            else parameter.dtype()
        )
        for parameter in _inputs(method)
    )
    return CallEntry(
        symbol=algorithm.mangled_name(method) + "__abi",
        abi_arguments=abi_arguments,
        operands=operands,
        return_type=_return_type(method),
        barrier_after=barrier,
        link_items=bundle.link_items,
        allocations=getattr(site.request, "allocations", ()),
        result=getattr(site.request, "result", None),
    )


# Rewriting --------------------------------------------------------------------


def _new_var(typemap, scope, name, ty, loc):
    var = scope.redefine(name, loc)
    typemap[var.name] = ty
    return var


class _StorageIR:
    """Emit typed array views and scalar indexing for a validated layout."""

    def __init__(self, typemap, calltypes, typingctx, scope, loc):
        self.typemap, self.calltypes = typemap, calltypes
        self.typingctx, self.scope, self.loc = typingctx, scope, loc
        self.body = []

    def assign(self, value, ty, name="$coop_storage", target=None):
        if target is None:
            target = _new_var(self.typemap, self.scope, name, ty, self.loc)
        self.body.append(nb.ir.Assign(value, target, self.loc))
        return target

    def constant(self, value):
        return self.assign(
            nb.ir.Const(value, self.loc), nb.types.IntegerLiteral(value)
        )

    def global_value(self, value, name):
        return self.assign(
            nb.ir.Global(name, value, self.loc),
            self.typingctx.resolve_value_type(value),
            name="$coop_" + name,
        )

    def attribute(self, value, name):
        return self.assign(
            nb.ir.Expr.getattr(value, name, self.loc),
            self.typingctx.resolve_getattr(self.typemap[value.name], name),
        )

    def call(self, function, args, kws=(), target=None):
        signature = self.typingctx.resolve_function_type(
            self.typemap[function.name],
            tuple(self.typemap[arg.name] for arg in args),
            {name: self.typemap[value.name] for name, value in kws},
        )
        expression = nb.ir.Expr.call(function, args, kws, self.loc)
        self.calltypes[expression] = signature
        return self.assign(expression, signature.return_type, target=target)

    def binary(self, function, left, right):
        signature = self.typingctx.resolve_function_type(
            function,
            (self.typemap[left.name], self.typemap[right.name]),
            {},
        )
        expression = nb.ir.Expr.binop(function, left, right, self.loc)
        self.calltypes[expression] = signature
        return self.assign(expression, signature.return_type)

    def byte_slice(self, backing, start, size):
        stop = self.binary(operator.add, start, self.constant(size))
        bounds = self.call(self.global_value(slice, "slice"), [start, stop])
        signature = self.typingctx.resolve_function_type(
            operator.getitem,
            (self.typemap[backing.name], self.typemap[bounds.name]),
            {},
        )
        expression = nb.ir.Expr.getitem(backing, bounds, self.loc)
        self.calltypes[expression] = signature
        return self.assign(
            expression, signature.return_type, name="$coop_region"
        )

    def linear_rank(self):
        cuda = self.global_value(nb.cuda, "cuda")
        thread = self.attribute(cuda, "threadIdx")
        block = self.attribute(cuda, "blockDim")
        x, y, z = (self.attribute(thread, axis) for axis in "xyz")
        dx, dy = (self.attribute(block, axis) for axis in "xy")
        return self.binary(
            operator.add,
            x,
            self.binary(
                operator.mul,
                dx,
                self.binary(operator.add, y, self.binary(operator.mul, dy, z)),
            ),
        )

    def instance_index(self, topology):
        from ..._core import SynchronizationScope

        rank = self.linear_rank()
        if topology.execution_scope is SynchronizationScope.NONE:
            return rank
        width = self.constant(topology.logical_width)
        if 32 % topology.logical_width:
            warp = self.binary(operator.floordiv, rank, self.constant(32))
            lane = self.binary(operator.mod, rank, self.constant(32))
            return self.binary(
                operator.add,
                self.binary(
                    operator.mul,
                    warp,
                    self.constant(32 // topology.logical_width),
                ),
                self.binary(operator.floordiv, lane, width),
            )
        return self.binary(operator.floordiv, rank, width)

    def warp_barrier(self, topology):
        mask = self.constant(0xFFFFFFFF)
        if topology.logical_width < 32:
            lane = self.binary(
                operator.and_, self.linear_rank(), self.constant(31)
            )
            width = self.constant(topology.logical_width)
            shift = self.binary(
                operator.mul, self.binary(operator.floordiv, lane, width), width
            )
            mask = self.binary(
                operator.lshift,
                self.constant((1 << topology.logical_width) - 1),
                shift,
            )
        sync = self.attribute(self.global_value(nb.cuda, "cuda"), "syncwarp")
        self.call(sync, [mask])


def _emit_backing(func_ir, typemap, calltypes, typingctx, storage):
    """Prepend the single shared allocation used by all validated slices."""
    entry = func_ir.blocks[min(func_ir.blocks)]
    emitter = _StorageIR(typemap, calltypes, typingctx, entry.scope, entry.loc)
    array = emitter.attribute(
        emitter.global_value(nb.shared, "shared"), "array"
    )
    backing = emitter.call(
        array,
        [
            emitter.constant(
                0 if storage.backing.kind == "dynamic" else storage.backing.size
            ),
            emitter.global_value(nb.types.uint8, "uint8"),
        ],
        (("alignment", emitter.constant(storage.backing.alignment)),),
    )
    entry.body[:0] = emitter.body
    return backing


def materialize(
    func_ir: Any,
    typemap: dict,
    calltypes: Any,
    typingctx: Any,
    sites: tuple[Site, ...],
    specialization: Specialization,
    storage: StorageLayout,
    bundle: Bundle,
) -> dict[str, CallEntry]:
    """Rewrite the IR for the resolved storage; return each call's entry."""

    from ..._core import SynchronizationScope
    from ._storage import ScratchSlice

    calls = {
        site.stmt.target.name: _call_entry(
            site,
            algorithm,
            scratch,
            scope is SynchronizationScope.BLOCK,
            bundle,
        )
        for site, algorithm, scratch, scope in zip(
            sites,
            specialization.site_algorithms,
            storage.scratch,
            storage.barrier_scopes,
            strict=True,
        )
    }
    from ._load_store import materialize_offsets

    materialize_offsets(
        func_ir, typemap, calltypes, typingctx, sites, specialization
    )
    if storage.backing is None:
        return calls
    backing = _emit_backing(func_ir, typemap, calltypes, typingctx, storage)
    scopes = {
        site.stmt.target.name: scope
        for site, scope in zip(
            sites,
            storage.barrier_scopes,
            strict=True,
        )
    }
    slices = dict(storage.slices)
    if not slices:
        # Preserve the stage-level layout API used by existing typed tests.
        region = storage.region
        assert region is not None
        slices = {
            site.stmt.target.name: ScratchSlice(
                region.offset,
                region.size,
                region.alignment,
                region.size,
                1,
            )
            for site, scratch in zip(sites, storage.scratch, strict=True)
            if scratch
        }
    for block in func_ir.blocks.values():
        rewritten = []
        for stmt in block.body:
            name = stmt.target.name if isinstance(stmt, nb.ir.Assign) else None
            region = slices.get(name) if name is not None else None
            if region is None:
                rewritten.append(stmt)
                continue
            emitter = _StorageIR(
                typemap, calltypes, typingctx, block.scope, stmt.loc
            )
            start = emitter.constant(region.offset)
            if region.instances > 1:
                index = emitter.instance_index(region.lowering_plan.topology)
                start = emitter.binary(
                    operator.add,
                    start,
                    emitter.binary(
                        operator.mul,
                        index,
                        emitter.constant(region.stride),
                    ),
                )
            view = emitter.byte_slice(backing, start, region.size)
            reservation = storage.reservations.get(name)
            if reservation is not None:
                emitter.call(
                    emitter.attribute(view, "view"),
                    [emitter.global_value(reservation.dtype, "reserve_dtype")],
                    target=stmt.target,
                )
                rewritten.extend(emitter.body)
                continue
            call = cast(nb.ir.Expr, stmt.value)
            call.kws = (*call.kws, (SCRATCH_KEYWORD, view))
            rewritten.extend(emitter.body)
            rewritten.append(stmt)
            if scopes[name] is SynchronizationScope.WARP:
                emitter.body = []
                emitter.warp_barrier(region.lowering_plan.topology)
                rewritten.extend(emitter.body)
        block.body = rewritten
    func_ir._definitions = nb.build_definitions(func_ir.blocks)
    return calls


__all__ = ["CallEntry", "CoopPlan", "materialize"]
