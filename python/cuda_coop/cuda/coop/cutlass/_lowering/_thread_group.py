# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lower group queries and synchronization to native CUDA extern calls.

Resolve hierarchy facts before emitting a request. Query wrappers return
CuTe integer values, membership returns Uint8, and synchronization
returns no value. All group and dtype choices are static; the extern ABI
has no runtime arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    cpp_level_expr,
    normalize_thread_level,
    render_hierarchy_decl,
    resolve_thread_group,
    validate_thread_group_query_dtype,
)

from .._compiler import _rendering, _state, _types
from .._compiler._launch import current_kernel_launch_facts
from .._thread_group import ThreadGroup

_SCOPE = "cuda.coop.cutlass"
_LEVEL_ORDER = {"thread": 0, "warp": 1, "block": 2, "cluster": 3, "grid": 4}
_QUERY_OPS = frozenset({"rank", "count"})
_SYNC_OPS = frozenset({"sync", "sync_aligned"})


@dataclass(frozen=True)
class _NativeGroupRequest:
    """Identify one group operation and its static query result type.

    Symbol names include group topology, operation, level, and dtype so
    distinct query contracts receive distinct definitions.
    """

    group: ThreadGroup
    op: str
    level: str = "thread"
    result_type: type | None = None
    kind: str = "native_group"

    def __post_init__(self):
        """Validate the operation and any requested integral query type."""

        if self.op not in _QUERY_OPS | _SYNC_OPS | {"is_member"}:
            raise ValueError(f"unsupported group operation {self.op!r}")
        if self.op in _QUERY_OPS:
            validate_thread_group_query_dtype(self.result_type, scope=_SCOPE)
            if self.result_type not in _types.INTEGER_VALUE_TYPES:
                raise TypeError(
                    "group query requires a supported integral dtype"
                )

    @property
    def symbol_name(self):
        parts = [
            "cuda_coop_cutlass_group",
            self.group.symbol_suffix,
            self.op,
        ]
        if self.op in _QUERY_OPS:
            parts.extend(
                (self.level, _types.TYPE_SPECIFICATIONS[self.result_type].token)
            )
        return "_".join(parts)


def _resolve_method_group(group, op, level="thread"):
    """Resolve the hierarchy needed by a query, membership test, or barrier.

    ThreadGroup method lowerings call this before generating native CUDA code.
    A descriptor can leave launch dimensions unknown; resolving it here
    supplies the compiler facts needed to interpret ranks and group sizes.

    Queries resolve the hierarchy through the queried level. A mapped group
    cannot query above its immediate parent. For synchronization, reject grid
    groups and mapped groups of warps. The latter need barrier storage with a
    lifetime that only the planner can own. Queries and membership tests on
    mapped groups of warps remain supported.

    Parameters
    ----------
    group : ThreadGroup
        Descriptor on which the user invoked a group method.
    op : str
        Normalized rank, count, membership, or synchronization operation.
    level : str
        Hierarchy level for rank/count queries. It determines how far up
        the launch hierarchy dimensions must be known.

    Returns
    -------
    ThreadGroup
        Resolved descriptor accepted by the requested method. Unsupported
        hierarchies or synchronization forms raise before code emission.
    """

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.ThreadGroup method requires a ThreadGroup")
    if (
        op in _QUERY_OPS
        and group.mapping is not None
        and (_LEVEL_ORDER[level] > _LEVEL_ORDER[group.mapping.parent])
    ):
        raise NotImplementedError(
            f"{_SCOPE} mapped ThreadGroup queries above the immediate parent "
            "require recursive group composition"
        )
    if op in _SYNC_OPS:
        if group.kind == "warps_within_block":
            raise NotImplementedError(
                f"{_SCOPE} mapped-Warp synchronization requires planner-owned "
                "barrier lifetime"
            )
        if group.kind == "grid":
            raise NotImplementedError(
                f"{_SCOPE} grid synchronization is unsupported"
            )
    return resolve_thread_group(
        group,
        current_kernel_launch_facts(),
        through_level=level if op in _QUERY_OPS else None,
    ).require_supported()


def _result_type(group, level, dtype):
    """Select a supported CuTe integer type for a hierarchy query.

    Defaults use Uint64 when the group or queried level is grid, and Uint32
    otherwise. Explicit dtype selectors, such as NumPy dtypes, become the
    matching CuTe integer type. The result is always a CuTe scalar.
    Non-integral dtypes are rejected.
    """

    if dtype is None:
        return (
            _types.Uint64
            if group.kind == "grid" or level == "grid"
            else _types.Uint32
        )
    dtype = _types.canonical_dsl_type(dtype)
    validate_thread_group_query_dtype(dtype, scope=_SCOPE)
    if dtype not in _types.INTEGER_VALUE_TYPES:
        raise TypeError(
            f"{_SCOPE} group query requires a supported integral dtype"
        )
    return dtype


def _native_query_expr(
    group: ThreadGroup, operation: str, unit: str, level: str
) -> str:
    """Query native CUDA levels using the known launch hierarchy."""

    hierarchy_arg = "" if group.hierarchy.implicit else ", hierarchy"
    return (
        f"{cpp_level_expr(unit)}.{operation}("
        f"{cpp_level_expr(level)}{hierarchy_arg})"
    )


def _query_expr(group: ThreadGroup, operation: str, level: str) -> str:
    """Query physical levels or the contiguous units of a mapped group."""

    if group.mapping is not None:
        mapping = group.mapping
        if _LEVEL_ORDER[level] > _LEVEL_ORDER[mapping.parent]:
            raise NotImplementedError(
                "mapped ThreadGroup queries above the immediate parent "
                "require recursive group composition"
            )
        rank = _native_query_expr(group, "rank", mapping.unit, mapping.parent)
        if level == mapping.parent:
            if operation == "rank":
                return f"({rank}) / {mapping.count}"
            count = _native_query_expr(
                group, "count", mapping.unit, mapping.parent
            )
            return f"({count}) / {mapping.count}"
        if level == mapping.unit:
            return (
                f"({rank}) % {mapping.count}"
                if operation == "rank"
                else str(mapping.count)
            )
        assert group.kind == "warps_within_block" and level == "thread"
        if operation == "rank":
            lane = _native_query_expr(group, "rank", "thread", "warp")
            return f"(({rank}) % {mapping.count}) * 32 + {lane}"
        return str(mapping.count * 32)

    if level == group.kind:
        return "0" if operation == "rank" else "1"
    if _LEVEL_ORDER[level] < _LEVEL_ORDER[group.kind]:
        return _native_query_expr(group, operation, level, group.kind)
    return _native_query_expr(group, operation, group.kind, level)


def _membership_expr(group: ThreadGroup) -> str:
    """Exclude only units outside the complete mapped prefix."""

    if group.mapping is None or group.mapping.exhaustive:
        return "true"
    mapping = group.mapping
    rank = _native_query_expr(group, "rank", mapping.unit, mapping.parent)
    count = _native_query_expr(group, "count", mapping.unit, mapping.parent)
    return f"({rank}) < (({count}) / {mapping.count}) * {mapping.count}"


def _sync_lines(group: ThreadGroup, operation: str) -> list[str]:
    """Emit the CUDA barrier for a supported group without owning scratch."""

    if group.kind == "thread":
        return []
    if group.kind == "warp":
        return ["  ::__syncwarp();"]
    if group.kind == "threads_within_warp":
        assert group.mapping is not None
        width = group.mapping.count
        lane = _native_query_expr(group, "rank", "thread", "warp")
        return [
            f"  if (!({_membership_expr(group)})) {{",
            "    return;",
            "  }",
            f"  const auto group_first_lane = ({lane} / {width}) * {width};",
            f"  ::__syncwarp({(1 << width) - 1}u << group_first_lane);",
        ]
    if group.kind == "block":
        return [
            "  ::__syncthreads();"
            if operation == "sync_aligned"
            else "  ::__barrier_sync(0);"
        ]
    if group.kind == "cluster":
        aligned = operation == "sync_aligned"
        barrier = (
            (
                (
                    '    asm volatile("barrier.cluster.arrive.aligned;"'
                    ' ::: "memory");'
                ),
                (
                    '    asm volatile("barrier.cluster.wait.aligned;"'
                    ' ::: "memory");'
                ),
            )
            if aligned
            else (
                "    ::__cluster_barrier_arrive();",
                "    ::__cluster_barrier_wait();",
            )
        )
        fallback = "::__syncthreads();" if aligned else "::__barrier_sync(0);"
        return [
            "  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, ({",
            *barrier,
            f"  }}), ({{ {fallback} }}))",
        ]
    raise NotImplementedError(
        f"cuda.coop.cutlass does not support {group.kind} synchronization"
    )


def _render_native_group(request):
    """Render a typed query, Uint8 membership test, or void barrier call."""

    group, op = request.group, request.op
    if op in _QUERY_OPS:
        cpp_type = _types.TYPE_SPECIFICATIONS[request.result_type].cpp_type
        return [
            f"{cpp_type} {request.symbol_name}() {{",
            *render_hierarchy_decl(group.hierarchy),
            (
                f"  return static_cast<{cpp_type}>"
                f"({_query_expr(group, op, request.level)});"
            ),
            "}",
        ]
    if op == "is_member":
        expression = _membership_expr(group)
        return [
            f"unsigned char {request.symbol_name}() {{",
            *render_hierarchy_decl(group.hierarchy),
            f"  return {expression} ? 1u : 0u;",
            "}",
        ]
    return [
        f"void {request.symbol_name}() {{",
        *render_hierarchy_decl(group.hierarchy),
        *_sync_lines(group, op),
        "}",
    ]


def _emit(request, result_type):
    """Queue a group definition and emit its zero-argument extern call.

    Convert a returned value to the declared CuTe type. On failure, restore
    queued request state without claiming to undo emitted IR.
    """

    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        value = ffi(
            name=request.symbol_name, params_types=[], return_type=result_type
        )()
        return None if result_type is None else result_type(value)
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise


def provider_group_query(*, group, op, level="thread", result_type=None):
    """Normalize a query level, resolve its group, and emit a typed value.

    ``ThreadGroup.rank`` and ``ThreadGroup.count`` call this while CuTe
    traces the kernel. The wrapper has no explicit operands: it reads the
    executing thread's built-in coordinates when the kernel runs.

    Parameters
    ----------
    group : ThreadGroup
        Group relative to which rank or count is requested.
    op : str
        Either rank or count.
    level : str
        Hierarchy unit to count or rank, such as thread, warp, or block.
    result_type : object or None
        Optional integral dtype selector. None uses Uint64 for grid queries
        and Uint32 for the other supported levels.

    Returns
    -------
    CuTe scalar
        Typed result of the generated native query call.
    """

    if op not in _QUERY_OPS:
        raise ValueError(f"unsupported group query {op!r}")
    level = normalize_thread_level(
        level, scope=_SCOPE, feature=f"ThreadGroup.{op}"
    )
    group = _resolve_method_group(group, op, level)
    dtype = _result_type(group, level, result_type)
    return _emit(_NativeGroupRequest(group, op, level, dtype), dtype)


def provider_group_sync(*, group, aligned):
    """Resolve a supported group and emit its selected barrier call."""

    op = "sync_aligned" if aligned else "sync"
    group = _resolve_method_group(group, op)
    _emit(_NativeGroupRequest(group, op), None)


def provider_group_membership(*, group):
    """Emit a Uint8 flag for membership in the resolved group."""

    group = _resolve_method_group(group, "is_member")
    return _emit(_NativeGroupRequest(group, "is_member"), _types.Uint8)


_rendering.register_bundle_renderer(
    "native_group",
    render=_render_native_group,
    include_lines=(
        "#include <cuda_runtime.h>",
        "#include <cuda/hierarchy>",
        "#include <cuda/std/cstdint>",
        "#include <nv/target>",
    ),
    cccl_headers=(("#include <cuda/hierarchy>", "cuda/hierarchy"),),
)
