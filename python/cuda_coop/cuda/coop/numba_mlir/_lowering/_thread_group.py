# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile group queries and synchronization helpers to device LTO IR.

A public group descriptor exists only during compilation. Its rank, count,
membership, and synchronization methods become C-ABI device helpers with the
launch hierarchy embedded in generated C++. Query helpers return scalar values;
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
    "#include <cuda_runtime.h>",
    "#include <cuda/hierarchy>",
    "#include <cuda/std/cstdint>",
    "#include <nv/target>",
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
        f"cuda.coop.numba_mlir does not support {group.kind} synchronization"
    )


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
            *render_hierarchy_decl(group.hierarchy),
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
            *render_hierarchy_decl(group.hierarchy),
            f"  return {_membership_expr(group)} ? 1u : 0u;",
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
            *render_hierarchy_decl(group.hierarchy),
            *_sync_lines(group, operation),
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


__all__ = [
    "_normalize_query_dtype",
    "make_group_method_invocable",
]
