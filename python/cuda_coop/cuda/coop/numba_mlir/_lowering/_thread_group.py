# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile group queries and synchronization helpers to device LTO IR.

A public group descriptor exists only during compilation. Its rank, count,
membership, and synchronization methods become C-ABI device helpers with the
group hierarchy embedded in generated C++. Query helpers return scalar values;
synchronization helpers return void. No runtime group object or TempStorage
pointer is passed through this ABI.
"""

from __future__ import annotations

import re
from typing import Any

import numba_cuda_mlir.numba_cuda.types as numba_types
from numba_cuda_mlir import cuda

from cuda.coop._core import (
    SynchronizationScope,
    ThreadGroup,
    cpp_level_expr,
    render_group_decl_lines,
    render_hierarchy_decl,
    validate_thread_group_query_dtype,
)

from .._compiler import _nvrtc
from .._compiler._operations import StorageABI
from .._compiler._parameters import normalize_dtype_param
from .._types import NUMBA_TYPES_TO_CPP, RawCAbiInvocable

_LEVEL_ORDER = {
    "thread": 0,
    "warp": 1,
    "block": 2,
    "cluster": 3,
    "grid": 4,
}
_INCLUDE_LINES = (
    "#define _CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX",
    "#define _CUDAX_DISABLE_CG_INTEROP",
    "#include <cuda/barrier>",
    "#include <cuda/devices>",
    "#include <cuda/hierarchy>",
    "#include <cuda/experimental/hierarchy.cuh>",
    "#include <cuda/std/cstdint>",
    "#include <cuda/std/type_traits>",
    "#include <cuda/experimental/coop/group>",
)


def _symbol_component(value: Any) -> str:
    component = re.sub(r"\W+", "_", str(value)).strip("_")
    return component or "anon"


def _cpp_type(dtype: Any) -> str:
    try:
        return NUMBA_TYPES_TO_CPP[dtype]
    except KeyError as exc:
        raise TypeError(
            "cuda.coop.numba_mlir group queries support built-in integral "
            f"dtypes; got {dtype!r}"
        ) from exc


def _current_cc() -> int:
    # The CUDA module reexports this accessor but omits it from its stub.
    device = (
        cuda.get_current_device()  # pyright: ignore[reportAttributeAccessIssue]
    )
    major, minor = device.compute_capability
    return int(major) * 10 + int(minor)


def _group_prelude(group: ThreadGroup) -> list[str]:
    """Render the group and any explicit hierarchy it depends on.

    Implicit hierarchies use the native current-group construction. Explicit
    hierarchies must be declared first so the group expression can refer to
    their known launch dimensions.
    """

    hierarchy = group.hierarchy
    if hierarchy.implicit:
        return render_group_decl_lines(group)
    return [
        *render_hierarchy_decl(hierarchy),
        *render_group_decl_lines(group),
    ]


def _mapped_warp_query_prelude(group: ThreadGroup) -> list[str]:
    """Render mapped-warp query metadata without a barrier group.

    Queries need only the physical warp rank, grouping count, and complete
    mapped prefix. Derive those values from the parent block instead of
    constructing a synchronization object with a separate barrier lifetime.
    """

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
            (
                f"  constexpr ::cuda::std::uint32_t group_warp_count = "
                f"{group.mapping.count};"
            ),
            (
                f"  constexpr ::cuda::std::uint32_t "
                f"grouped_warp_count = {grouped_warps};"
            ),
        )
    )
    return lines


def _query_expr(group: ThreadGroup, operation: str, level: str) -> str:
    """Select the C++ rank or count expression for the requested level.

    An inner level queries constituents; an outer level queries this group
    within its parent hierarchy. Same-level rank/count are zero/one. Mapped
    groups use their immediate parent and reject higher-level composition.

    Mapped physical warps use rank arithmetic from the query prelude so these
    reads do not construct a barrier-bearing group object.
    """

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


def _execution_scope(group: ThreadGroup) -> SynchronizationScope:
    """Map the descriptor to the execution scope of its native helper.

    This metadata describes participating threads. It does not request an
    extra scratch-reuse barrier after a query or synchronization helper.
    """

    return {
        "thread": SynchronizationScope.NONE,
        "warp": SynchronizationScope.WARP,
        "threads_within_warp": SynchronizationScope.WARP,
        "block": SynchronizationScope.BLOCK,
        "warps_within_block": SynchronizationScope.GROUP,
        "cluster": SynchronizationScope.GROUP,
        "grid": SynchronizationScope.GROUP,
    }[group.kind]


def _normalize_query_dtype(
    group: ThreadGroup,
    level: str,
    dtype: Any,
) -> Any:
    """Choose a default unsigned query type or validate an explicit integer.

    Use uint64 when the group or requested level is grid, otherwise uint32. An
    explicit dtype is normalized and checked against the shared query policy.
    The caller must choose enough width for its possible ranks or counts; this
    step does not prove that conversion is lossless.
    """

    if dtype is None:
        dtype = (
            numba_types.uint64
            if level == "grid" or group.kind == "grid"
            else numba_types.uint32
        )
    else:
        dtype = normalize_dtype_param(dtype)
    validate_thread_group_query_dtype(dtype, scope="cuda.coop.numba_mlir")
    return dtype


def make_group_method_invocable(
    *,
    group: ThreadGroup,
    operation: str,
    dtype: Any = None,
    level: str = "thread",
    compile_context: _nvrtc.CompileContext | None = None,
) -> RawCAbiInvocable:
    """Compile one rank, count, membership, or synchronization helper.

    The group planner calls this after validating a method call and
    reuses the result for equivalent calls in the same compilation
    attempt. The emitted helper replaces a descriptor method with a
    device function whose group is fixed in generated C++.

    Embed the resolved group and query controls in C++ and qualify its symbol
    with target and compiler context. Rank/count return the selected integer
    dtype; membership returns uint8; synchronization returns void. Incomplete
    mapped groups guard synchronization for excluded threads.

    Mapped physical-warp synchronization is rejected because its barrier
    lifetime is not managed here. The planner must resolve supported group and
    launch facts before calling this factory. Construction compiles LTO IR and
    returns a compiler-local ``RawCAbiInvocable`` with no operands.

    Parameters
    ----------
    group : ThreadGroup
        Supported descriptor already resolved against the kernel
        launch and requested hierarchy level.
    operation : str
        Native helper operation: rank, count, membership, or
        supported synchronization.
    dtype : object or None, optional
        Rank/count return dtype. ``None`` selects the query default;
        membership and synchronization use their fixed return forms.
    level : str, optional
        Compile-time hierarchy level for rank/count, defaulting to
        thread level.
    compile_context : nvrtc.CompileContext or None, optional
        Resolved toolkit/header inputs shared by the planner.
        ``None`` resolves them for this construction.
    """

    if not isinstance(group, ThreadGroup):
        raise TypeError("group must be a ThreadGroup")
    if group.kind == "warps_within_block" and operation in {
        "sync",
        "sync_aligned",
    }:
        raise NotImplementedError(
            "mapped-Warp synchronization requires "
            "planner-owned barrier lifetime"
        )
    if operation in {"rank", "count"}:
        dtype = _normalize_query_dtype(group, level, dtype)
    elif operation not in {"is_member", "sync", "sync_aligned"}:
        raise ValueError(f"unsupported group method operation {operation!r}")

    cc = _current_cc()
    if compile_context is None:
        compile_context = _nvrtc.resolve_compile_context()
    group_component = _symbol_component(group.symbol_suffix)
    if operation in {"rank", "count"}:
        cpp_type = _cpp_type(dtype)
        symbol = (
            "cuda_coop_numba_mlir_group_"
            f"{group_component}_{operation}_{level}_{_symbol_component(dtype)}_"
            f"cc{cc}_ctx_{compile_context.symbol_suffix}"
        )
        lines = [
            f'extern "C" __device__ {cpp_type} {symbol}() {{',
            *(
                _mapped_warp_query_prelude(group)
                if group.kind == "warps_within_block"
                else _group_prelude(group)
            ),
            f"  return static_cast<{cpp_type}>(",
            f"      {_query_expr(group, operation, level)});",
            "}",
        ]
        return_type = dtype
    elif operation == "is_member":
        symbol = (
            "cuda_coop_numba_mlir_group_"
            f"{group_component}_is_member_cc{cc}_ctx_"
            f"{compile_context.symbol_suffix}"
        )
        lines = [
            f'extern "C" __device__ ::cuda::std::uint8_t {symbol}() {{',
            *(
                _mapped_warp_query_prelude(group)
                if group.kind == "warps_within_block"
                else _group_prelude(group)
            ),
            (
                "  return group_warp_rank < grouped_warp_count ? 1u : 0u;"
                if group.kind == "warps_within_block"
                else "  return ::cuda::gpu_thread.is_part_of(group) ? 1u : 0u;"
            ),
            "}",
        ]
        return_type = numba_types.uint8
    else:
        symbol = (
            "cuda_coop_numba_mlir_group_"
            f"{group_component}_{operation}_cc{cc}_ctx_"
            f"{compile_context.symbol_suffix}"
        )
        lines = [
            f'extern "C" __device__ void {symbol}() {{',
            *_group_prelude(group),
            *(
                (
                    "  if (!::cuda::gpu_thread.is_part_of(group)) {",
                    "    return;",
                    "  }",
                )
                if group.mapping is not None
                and group.complete_membership is False
                else ()
            ),
            f"  group.{operation}();",
            "}",
        ]
        return_type = numba_types.void

    source = "\n".join((*_INCLUDE_LINES, "", *lines, ""))
    return RawCAbiInvocable(
        source=source,
        symbol=symbol,
        return_type=return_type,
        parameters=(),
        abi_transforms=(),
        cc=cc,
        compile_context=compile_context,
        storage_abi=StorageABI.NONE,
        execution_scope=_execution_scope(group),
        synchronization_scope=SynchronizationScope.NONE,
    )


__all__ = ["make_group_method_invocable"]
