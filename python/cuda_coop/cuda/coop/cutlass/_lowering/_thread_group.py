# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provider renderer for experimental cudax thread-group query helpers."""

from __future__ import annotations

import dataclasses
from typing import Any

from cutlass.cute.ffi import ffi

from .._compiler import _rendering, _state, _types
from .._thread_group import (
    ThreadGroup,
    cpp_level_expr,
    render_group_decl_lines,
    render_hierarchy_decl,
)

_provider_rendering = _rendering
_provider_state = _state
_provider_types = _types

_ROOT_SCOPE = "cuda.coop.cutlass"

_QUERY_OPS = frozenset({"rank", "count"})
_SYNC_OPS = frozenset({"sync", "sync_aligned"})
_MEMBERSHIP_OP = "is_member"
_LEVEL_ORDER = {
    "thread": 0,
    "warp": 1,
    "block": 2,
    "cluster": 3,
    "grid": 4,
}


@dataclasses.dataclass(frozen=True)
class _CudaxGroupRequest:
    group: ThreadGroup
    op: str
    level: str = "thread"
    result_type: type | None = None
    kind: str = "cudax_group"

    @property
    def symbol_name(self) -> str:
        parts = [
            "cuda_coop_cutlass_cudax_group",
            self.group.symbol_suffix,
            self.op,
        ]
        if self.op in _QUERY_OPS:
            parts.append(self.level)
            parts.append(_provider_types.TYPE_SPECS[self.result_type].token)
        return "_".join(parts)


def _result_type(result_type: type | None) -> type:
    if result_type is None:
        return _provider_types.Int32
    from cuda.coop._core import validate_thread_group_query_dtype

    validate_thread_group_query_dtype(result_type, scope=_ROOT_SCOPE)
    if result_type not in _provider_types.TYPE_SPECS:
        raise TypeError(
            f"{_ROOT_SCOPE}.ThreadGroup query dtype must be one of "
            f"{_provider_types.supported_names(_provider_types.ALL_PROVIDER_TYPES)}"
        )
    return result_type


def _validate_query(group: ThreadGroup, op: str, level: str) -> ThreadGroup:
    if op not in _QUERY_OPS:
        raise ValueError(f"Unsupported cudax group query op: {op}")
    from .._thread_group import _validate_query_launch

    return _validate_query_launch(
        group,
        feature=f"ThreadGroup.{op}",
        level=level,
    )


def _validate_sync(group: ThreadGroup, op: str) -> ThreadGroup:
    if op not in _SYNC_OPS:
        raise ValueError(f"Unsupported cudax group sync op: {op}")
    from .._thread_group import _validate_sync_launch

    return _validate_sync_launch(group, feature=f"ThreadGroup.{op}")


def _render_group_prelude(group: ThreadGroup) -> list[str]:
    if group.hierarchy.implicit:
        assert group.mapping is None
        hierarchy = "::cuda::experimental::implicit_hierarchy()"
        return [f"  ::cuda::experimental::this_{group.kind} group{{{hierarchy}}};"]
    return [
        *render_hierarchy_decl(group.hierarchy),
        *render_group_decl_lines(group),
    ]


def _render_query_expr(group: ThreadGroup, op: str, level: str) -> str:
    level_expr = cpp_level_expr(level)
    group_level = group.kind if group.mapping is None else group.mapping.parent
    if group.mapping is None and level == group_level:
        return "0" if op == "rank" else "1"
    if group.mapping is not None and level == group.mapping.parent:
        return f"group.{op}(group_parent)"
    if _LEVEL_ORDER[level] < _LEVEL_ORDER[group_level]:
        return f"{level_expr}.{op}(group)"
    return f"group.{op}({level_expr})"


def _render_cudax_group(request: _CudaxGroupRequest) -> list[str]:
    if request.op in _QUERY_OPS:
        spec = _provider_types.TYPE_SPECS[request.result_type]
        return [
            f"{spec.cpp_type} {request.symbol_name}() {{",
            *_render_group_prelude(request.group),
            (
                f"  return static_cast<{spec.cpp_type}>("
                f"{_render_query_expr(request.group, request.op, request.level)});"
            ),
            "}",
        ]
    if request.op in _SYNC_OPS:
        member = "sync_aligned" if request.op == "sync_aligned" else "sync"
        return [
            f"void {request.symbol_name}() {{",
            *_render_group_prelude(request.group),
            f"  group.{member}();",
            "}",
        ]
    if request.op == _MEMBERSHIP_OP:
        return [
            f"unsigned char {request.symbol_name}() {{",
            *_render_group_prelude(request.group),
            "  return ::cuda::gpu_thread.is_part_of(group) ? 1u : 0u;",
            "}",
        ]
    raise ValueError(f"Unsupported cudax group request op: {request.op}")


def _register_renderer() -> None:
    _provider_rendering.register_bundle_renderer(
        "cudax_group",
        render=_render_cudax_group,
        include_lines=(
            "#define _CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX",
            "#define _CUDAX_DISABLE_COOPERATIVE_GROUPS_INTEROP",
            "#include <cuda/barrier>",
            "#include <cuda/devices>",
            "#include <cuda/hierarchy>",
            "#include <cuda/std/type_traits>",
            "#include <cuda/experimental/group.cuh>",
        ),
        cccl_headers=(
            ("#include <cuda/experimental/group.cuh>", "cuda/experimental/group.cuh"),
        ),
    )


_register_renderer()


def provider_group_query(
    *,
    group: ThreadGroup,
    op: str,
    level: str = "thread",
    result_type: type | None = None,
) -> Any:
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_ROOT_SCOPE}.ThreadGroup query group must be a ThreadGroup")
    group = _validate_query(group, op, level)
    resolved_type = _result_type(result_type)
    request = _CudaxGroupRequest(
        group=group,
        op=op,
        level=level,
        result_type=resolved_type,
    )
    _provider_state.register_request(request)
    result = ffi(
        name=request.symbol_name,
        params_types=[],
        return_type=resolved_type,
    )()
    return _provider_state.remember_scalar_result_type(
        result,
        resolved_type,
        scope=_ROOT_SCOPE,
        compile_options_getter=lambda: _provider_state._get_cute_dsl().compile_options,
        group_metadata=_query_group_metadata(group),
    )


def provider_group_sync(
    *,
    group: ThreadGroup,
    aligned: bool,
) -> None:
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_ROOT_SCOPE}.ThreadGroup sync group must be a ThreadGroup")
    op = "sync_aligned" if aligned else "sync"
    group = _validate_sync(group, op)
    request = _CudaxGroupRequest(group=group, op=op)
    _provider_state.register_request(request)
    ffi(
        name=request.symbol_name,
        params_types=[],
        return_type=None,
    )()


def provider_group_membership(*, group: ThreadGroup) -> Any:
    if not isinstance(group, ThreadGroup):
        raise TypeError(
            f"{_ROOT_SCOPE}.ThreadGroup.is_member group must be a ThreadGroup"
        )
    from .._thread_group import _validate_membership_launch

    group = _validate_membership_launch(
        group,
        feature="ThreadGroup.is_member",
    )
    request = _CudaxGroupRequest(group=group, op=_MEMBERSHIP_OP)
    _provider_state.register_request(request)
    return ffi(
        name=request.symbol_name,
        params_types=[],
        return_type=_provider_types.Uint8,
    )()


def _query_group_metadata(group: ThreadGroup) -> Any:
    from cuda.coop._core import ResultVisibility

    from .._value_metadata import metadata_for_group

    return metadata_for_group(group, visibility=ResultVisibility.PER_MEMBER)


__all__ = [
    "_CudaxGroupRequest",
    "provider_group_membership",
    "provider_group_query",
    "provider_group_sync",
]
