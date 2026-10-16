# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native scratch descriptors and typed shared-array reservations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ..._core import StorageOwnership
from .._temp_storage import TempStorage
from . import _bridges as nb
from ._errors import TypedPipelineError

_PARAMETERS = ("size_in_bytes", "alignment", "auto_sync", "sharing")
_RESERVE = "coop.temp_storage.reserve"


class TempStorageType(nb.types.Type):
    """A descriptor contract with native constructor provenance."""

    def __init__(self, descriptor, origins=frozenset()):
        self.size_in_bytes = descriptor.size_in_bytes
        self.alignment = descriptor.alignment
        self.auto_sync = descriptor.auto_sync
        self.sharing = descriptor.sharing
        self.origins = frozenset(origins)
        super().__init__(name=f"coop.TempStorage{self.contract}")

    @property
    def contract(self):
        return (
            self.size_in_bytes,
            self.alignment,
            self.auto_sync,
            self.sharing,
        )

    @property
    def key(self):
        return self.contract, self.origins

    @property
    def semantics_kwargs(self):
        return {
            "storage_ownership": StorageOwnership.CALLER,
            "storage_sharing": self.sharing,
            "storage_size_in_bytes": self.size_in_bytes,
            "storage_alignment": self.alignment,
            "storage_auto_sync": self.auto_sync,
        }

    def with_origins(self, origins):
        return TempStorageType(self, origins)

    # Type.unify infers None, but the extension contract returns a merged type.
    def unify(self, typingctx, other) -> TempStorageType:  # pyright: ignore[reportIncompatibleMethodOverride]
        if not isinstance(other, TempStorageType):
            raise TypedPipelineError(
                "TempStorage variables must be bound to a TempStorage "
                "descriptor on every path"
            )
        if self.contract != other.contract:
            raise TypedPipelineError(
                "TempStorage aliases have inconsistent contracts"
            )
        origins = self.origins | other.origins
        if len(origins) > 1 and not self.auto_sync:
            raise TypedPipelineError(
                "TempStorage with auto_sync=False must be constructed at "
                "exactly one site; merging independent regions cannot "
                "preserve caller-managed synchronization"
            )
        return self.with_origins(origins)


def _constant(ty, name) -> Any:
    if isinstance(ty, nb.types.Literal):
        return ty.literal_value
    if isinstance(ty, nb.types.NoneType) or ty is None:
        return None
    if isinstance(ty, nb.types.Omitted):
        return ty.value
    raise TypedPipelineError(
        f"TempStorage {name} must be a compile-time literal"
    )


def _fold_constructor(args, kws):
    values = dict(zip(_PARAMETERS, args))
    values.update(dict(kws))
    return tuple(values.get(name) for name in _PARAMETERS)


def _constructor(self, args, kws):
    if len(args) > 1:
        raise TypedPipelineError(
            "TempStorage accepts only size_in_bytes positionally; "
            "alignment, auto_sync, and sharing are keyword-only"
        )
    if args and "size_in_bytes" in kws:
        raise TypedPipelineError(
            "TempStorage got multiple values for size_in_bytes"
        )
    if set(kws) - set(_PARAMETERS):
        raise TypedPipelineError("TempStorage got unexpected keyword arguments")
    bound = dict(zip(_PARAMETERS, args))
    bound.update(kws)
    requested = {
        _PARAMETERS.index(name)
        for name, ty in bound.items()
        if isinstance(ty, (nb.types.Integer, nb.types.Boolean))
        and not isinstance(ty, nb.types.Literal)
    }
    if requested:
        raise nb.force_literal_arg(requested, _fold_constructor)
    try:
        descriptor = TempStorage(
            **{name: _constant(ty, name) for name, ty in bound.items()}
        )
    except (TypeError, ValueError) as exc:
        raise TypedPipelineError(str(exc)) from None
    return nb.signature(TempStorageType(descriptor), *args, *kws.values())


def _reservation(owner, args, kws):
    if len(args) > 2:
        raise TypedPipelineError(
            "TempStorage.reserve accepts num_elems and dtype positionally; "
            "alignment is keyword-only"
        )
    names = ("num_elems", "dtype", "alignment")
    bound = dict(zip(names, args))
    if set(kws) - set(names) or bound.keys() & kws.keys():
        raise TypedPipelineError("TempStorage.reserve got invalid arguments")
    bound.update(kws)
    if "num_elems" not in bound or "dtype" not in bound:
        raise TypedPipelineError(
            "TempStorage.reserve requires num_elems and dtype"
        )
    count = bound["num_elems"]
    if (
        not isinstance(count, nb.types.IntegerLiteral)
        or count.literal_value <= 0
    ):
        raise TypedPipelineError(
            "TempStorage.reserve num_elems must be a compile-time "
            "positive integer"
        )
    dtype = bound["dtype"]
    try:
        if isinstance(dtype, nb.types.StringLiteral):
            from .._compiler._parameters import normalize_dtype_param

            dtype = normalize_dtype_param(dtype.literal_value)
        else:
            dtype = (
                dtype.dtype if isinstance(dtype, nb.types.DTypeSpec) else None
            )
        numpy_dtype = np.dtype(str(dtype))
        if dtype is None or numpy_dtype.kind not in "iufc":
            raise ValueError()
    except (TypeError, ValueError):
        raise TypedPipelineError(
            "TempStorage.reserve dtype must be a compile-time fixed-size "
            "integer, floating-point, or complex dtype"
        ) from None
    alignment = max(numpy_dtype.alignment, numpy_dtype.itemsize)
    if "alignment" in bound:
        message = (
            "TempStorage.reserve alignment must be a compile-time positive "
            "power of 2, or None"
        )
        try:
            requested = _constant(bound["alignment"], "reserve alignment")
        except TypedPipelineError:
            raise TypedPipelineError(message) from None
        if requested is not None:
            if (
                not isinstance(requested, int)
                or isinstance(requested, bool)
                or requested <= 0
                or requested & (requested - 1)
            ):
                raise TypedPipelineError(message)
            alignment = max(alignment, requested)
    if owner.auto_sync:
        raise TypedPipelineError(
            "TempStorage.reserve requires auto_sync=False; callers must "
            "synchronize accesses to reserved arrays explicitly"
        )
    return count.literal_value, dtype, numpy_dtype.itemsize, alignment


class _StorageAttributes(nb.AttributeTemplate):
    key = TempStorageType

    @nb.bound_function(_RESERVE)
    def resolve_reserve(self, owner, args, kws):
        _, dtype, _, _ = _reservation(owner, args, kws)
        return nb.signature(nb.types.Array(dtype, 1, "C"), *args, *kws.values())


@dataclass(frozen=True)
class Reservation:
    stmt: Any
    owner: tuple
    num_elems: int
    dtype: Any
    size_in_bytes: int
    alignment: int


@dataclass
class Descriptors:
    """Canonical local descriptor owners and their validated consumers."""

    constructors: dict[tuple, TempStorageType] = field(default_factory=dict)
    owners: dict[str, tuple] = field(default_factory=dict)
    reservations: dict[str, Reservation] = field(default_factory=dict)


def has_calls(typemap):
    return any(isinstance(ty, TempStorageType) for ty in typemap.values())


def analyze(func_ir, typemap, sites) -> Descriptors:
    """Validate descriptor ownership and escapes before provider compilation."""
    result = Descriptors()
    statements = [
        stmt
        for label in sorted(func_ir.blocks)
        for stmt in func_ir.blocks[label].body
    ]
    order = {
        stmt.target.name: index
        for index, stmt in enumerate(statements)
        if isinstance(stmt, nb.ir.Assign)
    }
    descriptor_types = [
        ty for ty in typemap.values() if isinstance(ty, TempStorageType)
    ]
    roots = {origin: origin for ty in descriptor_types for origin in ty.origins}

    def root(origin):
        while roots[origin] != origin:
            origin = roots[origin]
        return origin

    for ty in descriptor_types:
        if not ty.origins:
            raise TypedPipelineError(
                "temp_storage must originate from a TempStorage constructor "
                "in the compiled function"
            )
        owners = {root(origin) for origin in ty.origins}
        owner = min(owners, key=lambda origin: order.get(origin[1], 1 << 30))
        for origin in owners:
            roots[origin] = owner

    for name, ty in typemap.items():
        if isinstance(ty, TempStorageType):
            owner = root(next(iter(ty.origins)))
            result.owners[name] = owner
            result.constructors[owner] = ty
    result.constructors = dict(
        sorted(result.constructors.items(), key=lambda item: order[item[0][1]])
    )

    consumers = set()
    primitive_stmts = {id(site.stmt): site for site in sites}
    for stmt in statements:
        value = stmt.value if isinstance(stmt, nb.ir.Assign) else None
        used = {
            var.name
            for var in stmt.list_vars()
            if var.name in result.owners
            and (not isinstance(stmt, nb.ir.Assign) or var != stmt.target)
        }
        if isinstance(value, nb.ir.Expr) and value.op == "call":
            callee = typemap.get(value.func.name)
            if (
                isinstance(callee, nb.types.BoundFunction)
                and callee.typing_key == _RESERVE
            ):
                owner_type = callee.this
                owner = root(next(iter(owner_type.origins)))
                from ._inference import static_call_arguments

                args, kws = static_call_arguments(
                    func_ir,
                    value,
                    _RESERVE,
                    [typemap[arg.name] for arg in value.args],
                    {name: typemap[arg.name] for name, arg in value.kws},
                )
                count, dtype, itemsize, alignment = _reservation(
                    owner_type, args, kws
                )
                result.reservations[stmt.target.name] = Reservation(
                    stmt, owner, count, dtype, count * itemsize, alignment
                )
                consumers.add(owner)
                continue
        if not used:
            continue
        if isinstance(value, nb.ir.Var) or (
            isinstance(value, nb.ir.Expr)
            and value.op in {"cast", "phi", "static_getitem"}
        ):
            continue
        if (
            isinstance(value, nb.ir.Expr)
            and value.op == "getattr"
            and value.attr == "reserve"
        ):
            continue
        if id(stmt) in primitive_stmts:
            storage = dict(cast(nb.ir.Expr, value).kws).get("temp_storage")
            if storage is not None and used == {storage.name}:
                consumers.add(result.owners[storage.name])
                continue
        raise TypedPipelineError(
            "TempStorage values are opaque compile-time descriptors and "
            "may only be passed as temp_storage= to a cooperative primitive "
            "or used with reserve(); a descriptor would escape to runtime"
        )
    if result.constructors.keys() - consumers:
        raise TypedPipelineError(
            "TempStorage constructor has no consumer; pass it as "
            "temp_storage= or use reserve()"
        )
    return result


def _lower_descriptor(builder, target, args, kwargs):
    builder.store_var(
        target, nb.arith.constant(nb.IntegerType.get_signless(8), 0)
    )


class _DescriptorModel(nb.PrimitiveModel):
    def __init__(self, manager, fe_type):
        super().__init__(manager, fe_type, nb.IntegerType.get_signless(8))


_registered = False


def register():
    global _registered
    if _registered:
        return
    from ._inference import (
        constructor_origin_factories,
        numpy_integer_literal,
        operation_literal_arguments,
    )
    from ._templates import kernel_primitives, register_template

    for function, name in kernel_primitives().items():
        if name.rsplit(".", 1)[-1] != "TempStorage":
            continue
        register_template(
            function, _constructor, name.replace(".", "_"), prefer_literal=True
        )
        constructor_origin_factories[function] = lambda ty, origin: (
            ty.with_origins({origin})
        )
        operation_literal_arguments[function] = {
            name: numpy_integer_literal
            for name in (0, "size_in_bytes", "alignment")
        }
        nb.lowering_registry.lower(function, nb.types.VarArg(nb.types.Any))(
            _lower_descriptor
        )
    nb.register_model(TempStorageType)(_DescriptorModel)
    nb.typing_registry.register_attr(_StorageAttributes)
    operation_literal_arguments[_RESERVE] = {
        name: numpy_integer_literal for name in (0, "num_elems", "alignment")
    }
    _registered = True


__all__ = [
    "Descriptors",
    "Reservation",
    "TempStorageType",
    "analyze",
    "has_calls",
    "register",
]
