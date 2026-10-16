# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type group descriptors and validate partitions using the shared hierarchy.

Group values carry a compile-time request and lower to a placeholder byte.
After inference, partitions are checked against the exact launch dimensions.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from ..._core.group._resolution import resolve_thread_group
from ..._core.launch import LaunchFactOrigin, LaunchFacts
from ..._core.thread_group import ThreadGroup
from . import _bridges as nb
from ._errors import TypedPipelineError
from ._types import CoopGroupType

CONSTRUCTORS = frozenset(
    {"this_thread", "this_warp", "this_block", "this_cluster", "this_grid"}
)
_METHODS = ("group_by",)
_PREFIX = "coop.group."


def _constant(ty: Any, *, parameter: str) -> Any:
    if isinstance(ty, nb.types.Literal):
        return ty.literal_value
    if isinstance(ty, nb.types.Omitted):
        return ty.value
    if isinstance(ty, nb.types.NoneType) or ty is None:
        return None
    raise TypedPipelineError(
        f"ThreadGroup.{parameter} must be compile-time constant"
    )


def _bind(method: str, names: tuple[str, ...], args, kws) -> dict:
    if len(args) > len(names) or any(name not in names for name in kws):
        raise TypedPipelineError(f"invalid ThreadGroup.{method} arguments")
    bound = dict(zip(names, args))
    duplicates = bound.keys() & kws.keys()
    if duplicates:
        name = next(iter(duplicates))
        raise TypedPipelineError(
            f"ThreadGroup.{method} received {name} more than once"
        )
    bound.update(kws)
    return bound


def _partition(group: CoopGroupType, args, kws) -> CoopGroupType:
    if len(args) > 1:
        raise TypedPipelineError(
            "ThreadGroup.group_by exhaustive is keyword-only"
        )
    bound = _bind("group_by", ("count", "exhaustive"), args, kws)
    if "count" not in bound:
        raise TypedPipelineError("ThreadGroup.group_by requires count")
    count = _constant(bound["count"], parameter="group_by count")
    exhaustive = (
        _constant(bound["exhaustive"], parameter="group_by exhaustive")
        if "exhaustive" in bound
        else True
    )
    try:
        return CoopGroupType(group.group.group_by(count, exhaustive=exhaustive))
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise TypedPipelineError(str(exc)) from None


def _resolver(method):
    def resolve(self, group, args, kws):
        names = ("count", "exhaustive")
        bound = _bind(method, names, args, kws)
        requested = {
            index + 1
            for index, name in enumerate(names)
            if name in bound
            and isinstance(bound[name], (nb.types.Integer, nb.types.Boolean))
            and not isinstance(bound[name], nb.types.Literal)
        }
        if requested:
            raise nb.force_literal_arg(
                requested, partial(_fold_arguments, names=("group", *names))
            )
        result = _partition(group, args, kws)
        return nb.signature(result, *args, *kws.values())

    return nb.bound_function(_PREFIX + method)(resolve)


def _fold_arguments(args, kws, *, names):
    bound = dict(zip(names, args))
    bound.update(dict(kws))
    return tuple(bound.get(name) for name in names)


class _GroupAttributes(nb.AttributeTemplate):
    key = CoopGroupType


for _name in _METHODS:
    setattr(_GroupAttributes, "resolve_" + _name, _resolver(_name))


def _constructor(function, qualified_name):
    def generic(self, args, kws):
        if args or kws:
            raise TypedPipelineError(f"{qualified_name}() takes no arguments")
        return nb.signature(CoopGroupType(function()))

    return generic


def _calls(func_ir, typemap):
    for label in sorted(func_ir.blocks):
        for stmt in func_ir.blocks[label].body:
            if not (
                isinstance(stmt, nb.ir.Assign)
                and isinstance(stmt.value, nb.ir.Expr)
                and stmt.value.op == "call"
            ):
                continue
            ty = typemap.get(stmt.value.func.name)
            if isinstance(ty, nb.types.BoundFunction) and isinstance(
                ty.this, CoopGroupType
            ):
                method = str(ty.typing_key).removeprefix(_PREFIX)
                if method in _METHODS:
                    yield stmt, ty.this, method


def has_calls(func_ir: Any, typemap: dict) -> bool:
    """Whether group values require launch validation."""
    return any(isinstance(ty, CoopGroupType) for ty in typemap.values())


def launch_facts(launch: dict) -> LaunchFacts:
    """Preserve the default backend's verified launch-fact policy."""
    cluster = launch.get("cluster")
    names = ["exact_block_dim", "exact_grid_dim", "cluster_launch"]
    if cluster is not None:
        names.append("exact_cluster_dim")
    return LaunchFacts(
        exact_block_dim=launch.get("block"),
        exact_grid_dim=launch.get("grid"),
        exact_cluster_dim=cluster,
        cluster_launch=cluster is not None,
        cooperative_launch=False,
        provenance=tuple(
            LaunchFactOrigin(
                fact=name, source="numba_cuda_mlir_launch_config", verified=True
            )
            for name in names
        ),
    )


def resolve(
    state, func_ir, typemap, calltypes, *, target, context, launch
) -> None:
    """Validate group partitions once exact launch dimensions are available."""
    facts = launch_facts(launch)
    for block in func_ir.blocks.values():
        for stmt in block.body:
            if (
                isinstance(stmt, nb.ir.Assign)
                and isinstance(stmt.value, (nb.ir.Global, nb.ir.FreeVar))
                and isinstance(typemap.get(stmt.target.name), CoopGroupType)
            ):
                stmt.value = nb.ir.Const(0, stmt.loc)
    for stmt, group, method in _calls(func_ir, typemap):
        result = typemap[stmt.target.name]
        resolve_thread_group(result.group, facts).require_supported()


def _lower_descriptor(builder, target, args, kwargs):
    builder.store_var(
        target, nb.arith.constant(nb.IntegerType.get_signless(8), 0)
    )


_registered = False


def register() -> None:
    """Register native descriptor typing and lowering exactly once."""
    global _registered
    if _registered:
        return
    from ._templates import kernel_primitives, register_template

    any_args = nb.types.VarArg(nb.types.Any)
    nb.register_typeof(ThreadGroup)(lambda value, context: CoopGroupType(value))
    for obj, qualified_name in kernel_primitives().items():
        if qualified_name.rsplit(".", 1)[-1] in CONSTRUCTORS:
            register_template(
                obj,
                _constructor(obj, qualified_name),
                qualified_name.replace(".", "_"),
            )
            nb.lowering_registry.lower(obj)(_lower_descriptor)
    nb.typing_registry.register_attr(_GroupAttributes)
    for method in _METHODS:
        nb.lowering_registry.lower(_PREFIX + method, CoopGroupType, any_args)(
            _lower_descriptor
        )
    _registered = True


__all__ = ["CONSTRUCTORS", "has_calls", "launch_facts", "register", "resolve"]
