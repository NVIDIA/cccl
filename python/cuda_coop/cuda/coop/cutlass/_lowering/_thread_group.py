# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed CUDAX providers for exact-launch group queries and synchronization."""

from __future__ import annotations

from dataclasses import dataclass

from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    cpp_level_expr,
    normalize_thread_level,
    render_group_decl_lines,
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
class _CudaxGroupRequest:
    group: ThreadGroup
    op: str
    level: str = "thread"
    result_type: type | None = None
    kind: str = "cudax_group"

    def __post_init__(self):
        if self.op not in _QUERY_OPS | _SYNC_OPS | {"is_member"}:
            raise ValueError(f"unsupported group operation {self.op!r}")
        if self.op in _QUERY_OPS:
            validate_thread_group_query_dtype(self.result_type, scope=_SCOPE)
            if self.result_type not in _types.INTEGER_VALUE_TYPES:
                raise TypeError("group query requires a supported integral dtype")

    @property
    def symbol_name(self):
        parts = ["cuda_coop_cutlass_cudax_group", self.group.symbol_suffix, self.op]
        if self.op in _QUERY_OPS:
            parts.extend((self.level, _types.TYPE_SPECS[self.result_type].token))
        return "_".join(parts)


def _resolve_method_group(group, op, level="thread"):
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.ThreadGroup method requires a ThreadGroup")
    if op in _QUERY_OPS and group.mapping is not None:
        if _LEVEL_ORDER[level] > _LEVEL_ORDER[group.mapping.parent]:
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
            raise NotImplementedError(f"{_SCOPE} grid synchronization is unsupported")
    return resolve_thread_group(
        group,
        current_kernel_launch_facts(),
        through_level=level if op in _QUERY_OPS else None,
    ).require_supported()


def _result_type(group, level, dtype):
    if dtype is None:
        return (
            _types.Uint64 if group.kind == "grid" or level == "grid" else _types.Uint32
        )
    dtype = _types.canonical_dsl_type(dtype)
    validate_thread_group_query_dtype(dtype, scope=_SCOPE)
    if dtype not in _types.INTEGER_VALUE_TYPES:
        raise TypeError(f"{_SCOPE} group query requires a supported integral dtype")
    return dtype


def _group_prelude(group):
    if group.kind == "warps_within_block":
        return _mapped_warp_query_prelude(group)
    return [*render_hierarchy_decl(group.hierarchy), *render_group_decl_lines(group)]


def _mapped_warp_query_prelude(group: ThreadGroup) -> list[str]:
    """Render flat mapped-Warp metadata without constructing a barrier group."""

    assert group.kind == "warps_within_block"
    assert group.parent is not None
    assert group.mapping is not None
    hierarchy = group.hierarchy
    block_threads = hierarchy.block_thread_count
    assert block_threads is not None and block_threads % 32 == 0
    parent_warps = block_threads // 32
    grouped_warps = (parent_warps // group.mapping.count) * group.mapping.count
    lines = [] if hierarchy.implicit else render_hierarchy_decl(hierarchy)
    lines.extend(
        render_group_decl_lines(
            group.parent,
            var_name="group_parent",
        )
    )
    lines.extend(
        (
            "  auto group_warp_rank = ::cuda::warp.rank(group_parent);",
            f"  constexpr ::cuda::std::uint32_t group_warp_count = "
            f"{group.mapping.count};",
            f"  constexpr ::cuda::std::uint32_t grouped_warp_count = {grouped_warps};",
        )
    )
    return lines


def _query_expr(group: ThreadGroup, operation: str, level: str) -> str:
    if group.kind == "warps_within_block":
        assert group.mapping is not None
        block_threads = group.hierarchy.block_thread_count
        assert block_threads is not None and block_threads % 32 == 0
        parent_warps = block_threads // 32
        if level == "block":
            if operation == "rank":
                return "group_warp_rank / group_warp_count"
            return f"{parent_warps} / group_warp_count"
        if level == "warp":
            if operation == "rank":
                return "group_warp_rank % group_warp_count"
            return "group_warp_count"
        if level == "thread":
            if operation == "rank":
                return (
                    "(group_warp_rank % group_warp_count) * 32 + "
                    "::cuda::gpu_thread.rank(::cuda::warp, "
                    "group_parent.hierarchy())"
                )
            return "group_warp_count * 32"
        raise NotImplementedError(
            "mapped ThreadGroup queries above the immediate parent require "
            "recursive group composition"
        )

    level_expr = cpp_level_expr(level)
    if group.mapping is not None:
        parent_level = group.mapping.parent
        if level == parent_level:
            return f"group.{operation}(group_parent)"
        if _LEVEL_ORDER[level] > _LEVEL_ORDER[parent_level]:
            raise NotImplementedError(
                "mapped ThreadGroup queries above the immediate parent require "
                "recursive group composition"
            )
        return f"{level_expr}.{operation}(group)"

    if level == group.kind:
        return "0" if operation == "rank" else "1"
    if _LEVEL_ORDER[level] < _LEVEL_ORDER[group.kind]:
        return f"{level_expr}.{operation}(group)"
    return f"group.{operation}({level_expr})"


def _render_cudax_group(request):
    group, op = request.group, request.op
    if op in _QUERY_OPS:
        cpp_type = _types.TYPE_SPECS[request.result_type].cpp_type
        return [
            f"{cpp_type} {request.symbol_name}() {{",
            *_group_prelude(group),
            f"  return static_cast<{cpp_type}>({_query_expr(group, op, request.level)});",
            "}",
        ]
    if op == "is_member":
        expression = (
            "group_warp_rank < grouped_warp_count"
            if group.kind == "warps_within_block"
            else "::cuda::gpu_thread.is_part_of(group)"
        )
        return [
            f"unsigned char {request.symbol_name}() {{",
            *_group_prelude(group),
            f"  return {expression} ? 1u : 0u;",
            "}",
        ]
    return [
        f"void {request.symbol_name}() {{",
        *_group_prelude(group),
        f"  group.{op}();",
        "}",
    ]


def _emit(request, result_type):
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
    if op not in _QUERY_OPS:
        raise ValueError(f"unsupported group query {op!r}")
    level = normalize_thread_level(level, scope=_SCOPE, feature=f"ThreadGroup.{op}")
    group = _resolve_method_group(group, op, level)
    dtype = _result_type(group, level, result_type)
    return _emit(_CudaxGroupRequest(group, op, level, dtype), dtype)


def provider_group_sync(*, group, aligned):
    op = "sync_aligned" if aligned else "sync"
    group = _resolve_method_group(group, op)
    _emit(_CudaxGroupRequest(group, op), None)


def provider_group_membership(*, group):
    group = _resolve_method_group(group, "is_member")
    return _emit(_CudaxGroupRequest(group, "is_member"), _types.Uint8)


_rendering.register_bundle_renderer(
    "cudax_group",
    render=_render_cudax_group,
    include_lines=(
        "#define _CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX",
        "#define _CUDAX_DISABLE_CG_INTEROP",
        "#include <cuda/barrier>",
        "#include <cuda/devices>",
        "#include <cuda/hierarchy>",
        "#include <cuda/std/cstdint>",
        "#include <cuda/std/type_traits>",
        "#include <cuda/experimental/coop/group>",
    ),
    cccl_headers=(("cuda/experimental/coop/group", "cuda/experimental/coop/group"),),
)
