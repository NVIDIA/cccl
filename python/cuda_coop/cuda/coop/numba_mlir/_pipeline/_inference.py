# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative mutation contracts within Numba's native inference network.

Native constraints own ordinary expressions, calls and control-flow joins.
Allocation origins retain the distinction between inferred writes and typed
producers. Producer constraints establish those dtypes before indexed writes
or their consumers are typed; Store only supplies a dtype when writes do not.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

from . import _bridges as nb
from ._types import EXPLICIT, PRODUCER, STORE_DST, WRITE, ThreadDataType


@dataclass(frozen=True)
class DtypeSource:
    """Derive a mutation dtype from one typed operation operand."""

    operand: int | str
    resolve: Callable[[Any], Any]


@dataclass(frozen=True)
class OperationInference:
    """Output/source operand pairs for producer and Store contracts."""

    producers: tuple[tuple[int | str, int | str | DtypeSource], ...] = ()
    checked_producers: tuple[
        tuple[int | str, int | str | DtypeSource], ...
    ] = ()
    stores: tuple[tuple[int | str, int | str | DtypeSource], ...] = ()


operation_inference_contracts: dict[Callable, OperationInference] = {}
constructor_origin_factories: dict[Callable, Callable[[Any, tuple], Any]] = {}
operation_static_arguments: dict[
    Callable, dict[int | str, Callable[[Any], Any]]
] = {}
operation_literal_arguments: dict[
    Any, dict[int | str, Callable[[Any], Any]]
] = {}


def _argument(call, operand):
    if isinstance(operand, int):
        return call.args[operand] if operand < len(call.args) else None
    return dict(call.kws).get(operand)


def numpy_integer_literal(value):
    """Expose a captured NumPy integer as a call-local static control."""

    import numpy as np

    if isinstance(value, np.integer):
        return nb.types.IntegerLiteral(int(value))
    return None


def static_call_arguments(func_ir, call, typing_key, args, kws):
    """Normalize registered constant controls only at their call site."""

    controls = operation_literal_arguments.get(typing_key)
    if not controls:
        return args, kws
    args, kws = list(args), dict(kws)
    for operand, factory in controls.items():
        variable = _argument(call, operand)
        if variable is None:
            continue
        try:
            value = func_ir.infer_constant(variable)
        except nb.errors.ConstantInferenceError:
            continue
        normalized = factory(value)
        if normalized is not None:
            if isinstance(operand, int):
                args[operand] = normalized
            else:
                kws[operand] = normalized
    return tuple(args), kws


def _contains_imprecise_payload(value: Any) -> bool:
    if isinstance(value, ThreadDataType):
        return not value.is_precise()
    if isinstance(value, nb.types.BaseTuple):
        return any(
            _contains_imprecise_payload(item)
            for item in cast(Iterable[Any], value)
        )
    return False


def _native_type(value: Any) -> Any:
    """Keep inference-local allocation policy out of helper signatures."""

    if isinstance(value, ThreadDataType):
        return ThreadDataType(value.dtype, value.items, EXPLICIT)
    if isinstance(value, nb.types.BaseTuple):
        return nb.types.BaseTuple.from_types(
            tuple(_native_type(item) for item in cast(Iterable[Any], value))
        )
    return value


class _PayloadSetItem:
    """Let producers or checked writes establish the dtype before casting."""

    def __init__(self, native: Any) -> None:
        self.native = native
        self.loc = native.loc

    def __call__(self, infer: Any) -> None:
        payload = infer.variable_type(self.native.target)
        if isinstance(payload, ThreadDataType) and not payload.is_precise():
            return
        self.native(infer)

    def get_call_signature(self) -> Any:
        return self.native.get_call_signature()


class _PayloadWrite:
    """Apply write agreement to every possible write-inferred allocation."""

    def __init__(self, statement: Any) -> None:
        self.target = statement.target
        self.value = statement.value
        self.loc = statement.loc

    def __call__(self, infer: Any) -> None:
        payload = infer.variable_type(self.target)
        value = infer.variable_type(self.value)
        if not isinstance(payload, ThreadDataType) or value is None:
            return
        producers = infer.producer_origins()
        if producers is None:
            return
        # A separately compiled helper receives an already typed array.
        # Its ordinary writes cast to that dtype; only constructors owned
        # by this inference attempt participate in write-based inference.
        inferred = payload.inferred_origins & infer.payload_origins.keys()
        if not inferred - producers:
            return
        # Reads of inferred allocations depend on the context being solved.
        # They cannot fix another allocation before its independent writes
        # and Store destination have been considered. Once the target has a
        # dtype, these dependent writes use ordinary array casting.
        dependent = infer.inferred_read_origins() & infer.value_origins(
            self.value
        )
        if dependent and (
            payload.is_precise() or not infer.allow_dependent_writes
        ):
            return
        dtype = value.dtype if isinstance(value, nb.types.Array) else value
        dtype = nb.unliteral(dtype)
        if not dtype.is_precise():
            return
        if payload.is_precise() and payload.dtype != dtype:
            raise nb.errors.TypingError(
                "Failed to infer a consistent dtype from "
                "coop.ThreadData writes.",
                loc=self.loc,
            )
        if not payload.is_precise():
            infer.add_type(
                self.target.name, payload.with_dtype(dtype, WRITE), self.loc
            )


class _PayloadMutation:
    """Refine producer outputs, or validate/fill a Store payload dtype."""

    def __init__(
        self,
        payload: Any,
        source: Any,
        strength: int,
        loc: Any,
        resolver=None,
        checked=False,
    ):
        self.payload = payload
        self.source = source
        self.strength = strength
        self.loc = loc
        self.resolver = resolver
        self.checked = checked

    def __call__(self, infer: Any) -> None:
        payload = infer.variable_type(self.payload)
        source = infer.variable_type(self.source)
        if not isinstance(payload, ThreadDataType) or source is None:
            return
        dtype = (
            self.resolver(source)
            if self.resolver is not None
            else source.dtype
            if isinstance(source, nb.types.Array)
            else source
        )
        if dtype is None:
            return
        dtype = nb.unliteral(dtype)
        if not dtype.is_precise():
            return
        if self.strength == STORE_DST and not payload.is_precise():
            written = infer.write_origins()
            if (
                written is None or payload.origins & written
            ) and not infer.allow_fallback:
                return
        if payload.is_precise() and payload.dtype != dtype:
            raise nb.errors.TypingError(
                f"cooperative memory dtype {dtype} does not match "
                f"payload dtype {payload.dtype}",
                loc=self.loc,
            )
        infer.add_type(
            self.payload.name,
            payload.with_dtype(dtype, max(payload.strength, self.strength)),
            self.loc,
        )


class _CooperativeCall(nb.typeinfer.CallConstraint):
    def resolve(self, typeinfer, typevars, fnty):
        if isinstance(fnty, nb.types.Dispatcher):
            arguments = nb.typeinfer.fold_arg_vars(
                typevars, self.args, self.vararg, self.kws
            )
            if arguments is not None:
                args, kws = arguments
                if any(
                    _contains_imprecise_payload(arg)
                    for arg in (*args, *kws.values())
                ):
                    return
        previous = typeinfer.active_call
        typeinfer.active_call = self
        try:
            super().resolve(typeinfer, typevars, fnty)
        finally:
            typeinfer.active_call = previous
        if (
            self.target in typeinfer.payload_constructors
            or self.target in typeinfer.origin_factories
        ) and self.signature:
            self.signature = self.signature.replace(
                return_type=typevars[self.target].getone()
            )


class CooperativeTypeInferer(nb.typeinfer.TypeInferer):
    """Extend native constraints without re-inferring ordinary expressions."""

    def __init__(self, context, func_ir, warnings):
        super().__init__(context, func_ir, warnings)
        from ._templates import kernel_primitives

        constructors = {
            function
            for function, name in kernel_primitives().items()
            if name.rsplit(".", 1)[-1] == "ThreadData"
        }
        self.payload_constructors: set[str] = set()
        self.origin_factories: dict[str, Callable] = {}
        self.origin_scope = object()
        self.payload_origins: dict[tuple, ThreadDataType] = {}
        self.refining_origins: set[tuple] = set()
        self.payload_mutations: list[_PayloadMutation] = []
        self.payload_writes: list[_PayloadWrite] = []
        self.static_values: dict[int, Any] = {}
        self.static_arguments: dict[str, Any] = {}
        self.allow_fallback = False
        self.allow_dependent_writes = False
        self.active_call = None
        for block in func_ir.blocks.values():
            for statement in block.find_insts(nb.ir.Assign):
                call = statement.value
                if not isinstance(call, nb.ir.Expr) or call.op != "call":
                    continue
                try:
                    function = func_ir.infer_constant(call.func)
                except nb.errors.ConstantInferenceError:
                    continue
                if function in constructors:
                    self.payload_constructors.add(statement.target.name)
                factory = constructor_origin_factories.get(function)
                if factory is not None:
                    self.origin_factories[statement.target.name] = factory
                for operand, factory in operation_static_arguments.get(
                    function, {}
                ).items():
                    variable = _argument(call, operand)
                    if variable is None:
                        continue
                    try:
                        value = func_ir.infer_constant(variable)
                    except nb.errors.ConstantInferenceError:
                        continue
                    try:
                        context.resolve_value_type(value)
                    except ValueError:
                        static_type = factory(value)
                        self.static_values[id(value)] = static_type
                        self.static_arguments[variable.name] = static_type
                contract = operation_inference_contracts.get(function)
                if contract is None:
                    continue
                for pairs, strength, checked in (
                    (contract.producers, PRODUCER, False),
                    (contract.checked_producers, PRODUCER, True),
                    (contract.stores, STORE_DST, False),
                ):
                    for output, source in pairs:
                        resolver = None
                        if isinstance(source, DtypeSource):
                            resolver = source.resolve
                            source = source.operand
                        target = _argument(call, output)
                        value = _argument(call, source)
                        if target is None or value is None:
                            continue
                        self.payload_mutations.append(
                            _PayloadMutation(
                                target,
                                value,
                                strength,
                                statement.loc,
                                resolver,
                                checked,
                            )
                        )

    def resolve_value_type(self, inst, val):
        static_type = self.static_values.get(id(val))
        if static_type is not None:
            return static_type
        return super().resolve_value_type(inst, val)

    def typeof_assign(self, inst):
        static_type = self.static_arguments.get(inst.target.name)
        if static_type is not None:
            self.add_type(inst.target.name, static_type, inst.loc)
            return
        super().typeof_assign(inst)

    def variable_type(self, variable):
        value = self.typevars[variable.name]
        return value.getone() if value.defined else None

    def producer_origins(self):
        origins = set()
        for mutation in self.payload_mutations:
            if mutation.strength != PRODUCER or mutation.checked:
                continue
            payload = self.variable_type(mutation.payload)
            if payload is None:
                return None
            if isinstance(payload, ThreadDataType):
                origins.update(payload.origins)
        return origins

    def inferred_read_origins(self):
        """Allocations whose reads depend on the dtype being inferred."""

        inferred = {
            origin
            for origin, declared in self.payload_origins.items()
            if origin in declared.inferred_origins
        }
        return inferred - (self.producer_origins() or set())

    def value_origins(self, value):
        """Read payload dependencies without evaluating ordinary dtypes.

        Native SSA definitions retain aliases, phis and expression inputs.
        A helper's payload arguments count conservatively as dependencies;
        its native call constraint still owns the return type.
        """

        origins, pending, seen = set(), [value], set()
        while pending:
            variable = pending.pop()
            if variable.name in seen:
                continue
            seen.add(variable.name)
            ty = self.variable_type(variable)
            if isinstance(ty, ThreadDataType):
                origins.update(ty.origins)
                continue
            for definition in self.func_ir._definitions.get(variable.name, ()):
                if isinstance(definition, nb.ir.Var):
                    pending.append(definition)
                elif isinstance(definition, nb.ir.Expr):
                    pending.extend(definition.list_vars())
        return origins

    def propagate(self, raise_errors=True):
        """Exhaust independent evidence before supplying fallback context."""

        errors = super().propagate(raise_errors=False)
        pending = any(
            mutation.strength == STORE_DST
            and isinstance(
                payload := self.variable_type(mutation.payload), ThreadDataType
            )
            and not payload.is_precise()
            for mutation in self.payload_mutations
        )
        if pending:
            self.allow_fallback = True
            errors = super().propagate(raise_errors=False)
        if (not self.payload_origins or not self.payload_writes) and (
            not errors or not raise_errors
        ):
            return errors
        self.allow_dependent_writes = True
        return super().propagate(raise_errors=raise_errors)

    def write_origins(self):
        origins = set()
        for write in self.payload_writes:
            payload = self.variable_type(write.target)
            if payload is None:
                return None
            if isinstance(payload, ThreadDataType):
                origins.update(payload.origins)
        return origins

    def add_type(self, var, tp, loc, unless_locked=False):
        factory = self.origin_factories.get(var)
        if factory is not None:
            tp = factory(tp, (self.origin_scope, var))
        if isinstance(tp, ThreadDataType):
            if var in self.payload_constructors:
                origin = (self.origin_scope, var)
                origins = frozenset({origin})
                inferred = (
                    frozenset()
                    if tp.is_precise() and tp.strength == EXPLICIT
                    else origins
                )
                tp = tp.with_origins(origins, inferred)
                self.payload_origins.setdefault(origin, tp)
            if tp.is_precise():
                for origin in tp.origins:
                    declared = self.payload_origins.get(origin)
                    if declared is None or origin in self.refining_origins:
                        continue
                    self.refining_origins.add(origin)
                    try:
                        super().add_type(
                            origin[1],
                            declared.with_dtype(
                                tp.dtype, max(declared.strength, tp.strength)
                            ),
                            loc,
                        )
                    finally:
                        self.refining_origins.remove(origin)
        super().add_type(var, tp, loc, unless_locked)

    def copy_type(self, src_var, dest_var, loc):
        super().copy_type(src_var, dest_var, loc)
        result = self.typevars[dest_var]
        if result.defined and isinstance(result.getone(), ThreadDataType):
            self.add_type(dest_var, result.getone(), loc)

    def typeof_call(self, inst, target, call):
        constraint = _CooperativeCall(
            target.name,
            call.func.name,
            call.args,
            call.kws,
            call.vararg,
            loc=inst.loc,
        )
        self.constraints.append(constraint)
        self.calls.append((inst.value, constraint))

    def resolve_call(self, fnty, pos_args, kw_args):
        if self.active_call is not None:
            pos_args, kw_args = static_call_arguments(
                self.func_ir,
                self.active_call,
                getattr(fnty, "typing_key", None),
                pos_args,
                kw_args,
            )
        if isinstance(fnty, nb.types.Dispatcher):
            pos_args = tuple(_native_type(arg) for arg in pos_args)
            kw_args = {name: _native_type(arg) for name, arg in kw_args.items()}
        result = super().resolve_call(fnty, pos_args, kw_args)
        if (
            result is not None
            and isinstance(result.return_type, nb.types.IteratorType)
            and any(
                _contains_imprecise_payload(arg)
                for arg in (*pos_args, *kw_args.values())
            )
        ):
            return None
        return result

    def typeof_setitem(self, inst):
        native = nb.typeinfer.SetItemConstraint(
            inst.target, inst.index, inst.value, inst.loc
        )
        constraint = _PayloadSetItem(native)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))
        self.payload_writes.append(_PayloadWrite(inst))

    def typeof_static_setitem(self, inst):
        native = nb.typeinfer.StaticSetItemConstraint(
            inst.target, inst.index, inst.index_var, inst.value, inst.loc
        )
        constraint = _PayloadSetItem(native)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))
        self.payload_writes.append(_PayloadWrite(inst))

    def build_constraint(self):
        super().build_constraint()
        for mutation in self.payload_mutations:
            self.constraints.append(mutation)
        for write in self.payload_writes:
            self.constraints.append(write)


__all__ = [
    "CooperativeTypeInferer",
    "DtypeSource",
    "OperationInference",
    "constructor_origin_factories",
    "numpy_integer_literal",
    "operation_inference_contracts",
    "operation_literal_arguments",
    "operation_static_arguments",
    "static_call_arguments",
]
