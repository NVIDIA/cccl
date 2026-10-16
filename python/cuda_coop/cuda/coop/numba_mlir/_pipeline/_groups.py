# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type group descriptors and lower native queries using the shared hierarchy.

Group values carry a compile-time request and lower to a placeholder byte.
After inference, query and synchronization calls resolve against the launch
and reuse the existing native group-helper provider. No CUDA rank formula or
synchronization policy is reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

from ..._core.group._resolution import resolve_thread_group
from ..._core.launch import LaunchFactOrigin, LaunchFacts
from ..._core.thread_group import ThreadGroup, normalize_thread_level
from .._lowering._thread_group import (
    _normalize_query_dtype,
    make_group_method_invocable,
)
from .._types import kernel_compile_target
from . import _bridges as nb
from ._errors import TypedPipelineError
from ._types import CoopGroupType

CONSTRUCTORS = frozenset(
    {"this_thread", "this_warp", "this_block", "this_cluster", "this_grid"}
)
_METHODS = (
    "group_by",
    "rank",
    "count",
    "rank_as",
    "count_as",
    "is_member",
    "sync",
    "sync_aligned",
)
_PREFIX = "coop.group."
_LEVELS = {"thread": 0, "warp": 1, "block": 2, "cluster": 3, "grid": 4}


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


def _dtype(ty: Any) -> Any:
    if ty is None or isinstance(ty, (nb.types.NoneType, nb.types.Omitted)):
        return None
    if isinstance(ty, (nb.types.NumberClass, nb.types.DTypeSpec)):
        return ty.dtype
    if isinstance(ty, nb.types.Function) and ty.typing_key is int:
        return nb.types.int64
    raise TypedPipelineError(
        "ThreadGroup query dtype must be a compile-time integral dtype"
    )


@dataclass(frozen=True)
class Method:
    operation: str
    level: str
    dtype: Any


def _method(group: CoopGroupType, method: str, args, kws) -> Method:
    if method in {"rank", "count", "rank_as", "count_as"}:
        names = ("dtype", "level") if method.endswith("_as") else ("level",)
        bound = _bind(method, names, args, kws)
        level = (
            _constant(bound["level"], parameter=f"{method} level")
            if "level" in bound
            else "thread"
        )
        operation = method.removesuffix("_as")
        try:
            level = normalize_thread_level(
                level,
                scope="cuda.coop.numba_mlir",
                feature=f"ThreadGroup.{operation}",
            )
            mapping = group.group.mapping
            if mapping is not None and _LEVELS[level] > _LEVELS[mapping.parent]:
                raise NotImplementedError(
                    "mapped ThreadGroup queries above the immediate parent "
                    "require recursive group composition"
                )
            dtype = _normalize_query_dtype(
                group.group, level, _dtype(bound.get("dtype"))
            )
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise TypedPipelineError(str(exc)) from None
        return Method(operation, level, dtype)
    _bind(method, (), args, kws)
    if method in {"sync", "sync_aligned"}:
        if group.group.kind == "warps_within_block":
            raise TypedPipelineError(
                "mapped-Warp synchronization requires owned barrier lifetime"
            )
        if group.group.kind == "grid":
            raise TypedPipelineError(
                "grid synchronization requires a verified cooperative launch, "
                "which the current launch descriptor cannot request"
            )
        dtype = nb.types.void
    else:
        dtype = nb.types.uint8
    return Method(method, "thread", dtype)


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
        if method == "group_by":
            names = ("count", "exhaustive")
        elif method in {"rank", "count", "rank_as", "count_as"}:
            names = ("dtype", "level") if method.endswith("_as") else ("level",)
        else:
            names = ()
        bound = _bind(method, names, args, kws)
        requested = {
            index + 1
            for index, name in enumerate(names)
            if name != "dtype"
            and name in bound
            and isinstance(
                bound[name],
                (nb.types.Integer, nb.types.Boolean, nb.types.UnicodeType),
            )
            and not isinstance(bound[name], nb.types.Literal)
        }
        if requested:
            raise nb.force_literal_arg(
                requested, partial(_fold_arguments, names=("group", *names))
            )
        if method == "group_by":
            result = _partition(group, args, kws)
        else:
            result = _method(group, method, args, kws).dtype
            if result == nb.types.void:
                result = nb.types.none
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
    """Whether group values require launch validation or native calls."""
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


@dataclass(frozen=True)
class NativeCall:
    symbol: str
    return_type: Any
    link_item: Any


def resolve(
    state, func_ir, typemap, calltypes, *, target, context, launch
) -> None:
    """Resolve native method providers once per kernel and record lowering."""
    facts = launch_facts(launch)
    calls = {}
    providers = {}
    for block in func_ir.blocks.values():
        for stmt in block.body:
            if (
                isinstance(stmt, nb.ir.Assign)
                and isinstance(stmt.value, (nb.ir.Global, nb.ir.FreeVar))
                and isinstance(typemap.get(stmt.target.name), CoopGroupType)
            ):
                stmt.value = nb.ir.Const(0, stmt.loc)
    with kernel_compile_target(target):
        for stmt, group, method in _calls(func_ir, typemap):
            if method == "group_by":
                # Its receiver has already been checked by typing. Validate
                # the resulting partition once exact dimensions are available.
                result = typemap[stmt.target.name]
                resolve_thread_group(result.group, facts).require_supported()
                continue
            expr = cast(nb.ir.Expr, stmt.value)
            request = _method(
                group,
                method,
                [typemap[a.name] for a in expr.args],
                {k: typemap[v.name] for k, v in expr.kws},
            )
            through = (
                request.level
                if request.operation in {"rank", "count"}
                else None
            )
            resolved = resolve_thread_group(
                group.group, facts, through_level=through
            ).require_supported()
            key = (
                resolved.semantic_key,
                request.operation,
                request.level,
                request.dtype,
            )
            native = providers.get(key)
            if native is None:
                provider = make_group_method_invocable(
                    group=resolved,
                    operation=request.operation,
                    dtype=request.dtype,
                    level=request.level,
                    compile_context=context,
                )
                image = Path(provider.files[0]).read_bytes()
                native = NativeCall(
                    provider.symbol,
                    provider.return_type,
                    nb.LTOIR(lambda image=image: image, name=provider.symbol),
                )
                providers[key] = native
            calls[stmt.target.name] = native
    state.metadata["coop_group_calls"] = calls


def _lower_descriptor(builder, target, args, kwargs):
    builder.store_var(
        target, nb.arith.constant(nb.IntegerType.get_signless(8), 0)
    )


def _lower_method(builder, target, args, kwargs):
    native = builder.metadata.get("coop_group_calls", {}).get(target.name)
    if native is None:
        raise AssertionError(
            "native group method reached lowering without resolution"
        )
    returns = native.return_type not in {nb.types.none, nb.types.void}
    result_types = (
        [builder.get_return_type(native.return_type)] if returns else []
    )
    function_type = nb.FunctionType.get(inputs=[], results=result_types)
    callee = nb.get_or_insert_function(
        native.symbol, function_type, builder.mlir_gpu_module
    )
    result = nb.func.call(
        result=result_types, callee=callee.name.value, operands_=[]
    )
    if returns:
        builder.store_var(
            target, builder.from_return(native.return_type, result)
        )
    builder.link_external_item(native.link_item)


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
            _lower_descriptor if method == "group_by" else _lower_method
        )
    _registered = True


__all__ = ["CONSTRUCTORS", "has_calls", "launch_facts", "register", "resolve"]
