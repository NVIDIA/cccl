# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LTO-IR providers for Numba-CUDA-MLIR CUDAX group operations."""

from __future__ import annotations

import os
import re
import weakref
from typing import Any

from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.descriptor import mlir_target
from numba_cuda_mlir.extending import overload, typing_registry
from numba_cuda_mlir.types import signature

from cuda.coop._core import (
    ThreadGroup,
    cpp_level_expr,
    render_group_decl_lines,
    render_hierarchy_decl,
    validate_thread_group_query_dtype,
)

from .._compiler import _nvrtc
from .._compiler._artifacts import make_binary_tempfile
from .._compiler._parameters import normalize_dtype_param
from .._types import (
    NUMBA_TYPES_TO_CPP,
    war_introspection,
    war_introspection_call_with_transforms,
)

_LEVEL_ORDER = {
    "thread": 0,
    "warp": 1,
    "block": 2,
    "cluster": 3,
    "grid": 4,
}
_INCLUDE_LINES = (
    "#define _CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX",
    "#define _CUDAX_DISABLE_COOPERATIVE_GROUPS_INTEROP",
    "#include <cuda/barrier>",
    "#include <cuda/devices>",
    "#include <cuda/functional>",
    "#include <cuda/hierarchy>",
    "#include <cuda/std/cstdint>",
    "#include <cuda/std/functional>",
    "#include <cuda/std/type_traits>",
    "#include <cuda/experimental/coop.cuh>",
    "#include <cuda/experimental/group.cuh>",
)
_GROUP_METHOD_INVOCABLE_CACHE: dict[tuple[Any, ...], "_RawCAbiInvocable"] = {}


def _symbol_component(value: Any) -> str:
    component = re.sub(r"\W+", "_", str(value)).strip("_")
    return component or "anon"


def _type_token(dtype: Any) -> str:
    return _symbol_component(str(dtype))


def _cpp_type(dtype: Any) -> str:
    dtype = normalize_dtype_param(dtype)
    try:
        return NUMBA_TYPES_TO_CPP[dtype]
    except KeyError as exc:
        raise TypeError(
            "cuda.coop.numba_mlir group providers currently support built-in "
            f"scalar dtypes; got {dtype!r}"
        ) from exc


def _current_cc() -> int:
    major, minor = cuda.get_current_device().compute_capability
    return int(major) * 10 + int(minor)


def _compile_ltoir(
    source: str,
    *,
    cc: int,
    context: _nvrtc.CompileContext,
) -> bytes:
    _, image = _nvrtc.compile(
        cpp=source,
        cc=cc,
        rdc=True,
        code="lto",
        context=context,
    )
    return bytes(image)


def _cleanup_temp_file(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class _RawCAbiInvocable:
    """One always-inlined, C-ABI device helper backed by real LTO-IR."""

    def __init__(
        self,
        *,
        source: str,
        symbol: str,
        return_type: Any,
        cc: int,
        compile_context: _nvrtc.CompileContext,
        expected_types: tuple[Any, ...] = (),
        abi_types: tuple[Any, ...] = (),
        transforms: tuple[str, ...] = (),
    ) -> None:
        if not (len(expected_types) == len(abi_types) == len(transforms)):
            raise ValueError("group provider ABI metadata has inconsistent arity")

        self._temp_file = make_binary_tempfile(
            _compile_ltoir(source, cc=cc, context=compile_context),
            ".ltoir",
        )
        self._temp_file_finalizer = weakref.finalize(
            self,
            _cleanup_temp_file,
            self._temp_file.name,
        )
        self.specialization = None

        extern_fn = cuda.declare_device(
            symbol,
            signature(return_type, *abi_types),
            link=self.files,
            abi="c",
        )

        def invocable_impl(*actual_types):
            if len(actual_types) != len(expected_types):
                return None
            typingctx = mlir_target.typing_context
            if any(
                typingctx.can_convert(actual, expected) is None
                for actual, expected in zip(actual_types, expected_types)
            ):
                return None
            impl = war_introspection_call_with_transforms(
                extern_fn,
                transforms,
                returns_value=return_type is not types.void,
            )
            setattr(impl, "__numba_cuda_mlir_link__", self.files)
            return impl

        wrapped = war_introspection(invocable_impl, len(expected_types))
        setattr(wrapped, "__numba_cuda_mlir_link__", self.files)
        overload(self, inline="always", typing_registry=typing_registry)(wrapped)

    @property
    def files(self) -> list[str]:
        return [self._temp_file.name]

    @property
    def temp_storage_bytes(self) -> int:
        return 0

    @property
    def temp_storage_alignment(self) -> int:
        return 1

    def __call__(self, *args: Any) -> Any:
        del args
        raise RuntimeError(
            "group provider invocables may only be called inside a "
            "numba_cuda_mlir.cuda.jit kernel"
        )


def _source(lines: list[str]) -> str:
    return "\n".join((*_INCLUDE_LINES, "", *lines, ""))


def _group_prelude(group: ThreadGroup) -> list[str]:
    assert group.hierarchy is not None
    if group.hierarchy.implicit:
        assert group.mapping is None
        hierarchy = "::cuda::experimental::implicit_hierarchy()"
        return [f"  ::cuda::experimental::this_{group.kind} group{{{hierarchy}}};"]
    return [
        *render_hierarchy_decl(group.hierarchy),
        *render_group_decl_lines(group),
    ]


def _query_expr(group: ThreadGroup, operation: str, level: str) -> str:
    level_expr = cpp_level_expr(level)
    group_level = group.kind if group.mapping is None else group.mapping.parent
    if group.mapping is None and level == group_level:
        return "0" if operation == "rank" else "1"
    if group.mapping is not None and level == group.mapping.parent:
        return f"group.{operation}(group_parent)"
    if _LEVEL_ORDER[level] < _LEVEL_ORDER[group_level]:
        return f"{level_expr}.{operation}(group)"
    return f"group.{operation}({level_expr})"


def make_group_method_invocable(
    *,
    group: ThreadGroup,
    operation: str,
    dtype: Any = None,
    level: str = "thread",
    compile_context: _nvrtc.CompileContext | None = None,
) -> _RawCAbiInvocable:
    """Materialize a query, membership, or synchronization helper."""

    dtype = types.int32 if dtype is None else normalize_dtype_param(dtype)
    if operation in {"rank", "count"}:
        validate_thread_group_query_dtype(dtype, scope="cuda.coop.numba_mlir")
    cc = _current_cc()
    if compile_context is None:
        compile_context = _nvrtc.resolve_compile_context()
    key = (
        "method",
        group.semantic_key,
        operation,
        dtype,
        level,
        cc,
        compile_context,
    )
    cached = _GROUP_METHOD_INVOCABLE_CACHE.get(key)
    if cached is not None:
        return cached

    stem = (
        "cuda_coop_numba_mlir_group_"
        f"{group.symbol_suffix}_{operation}_{level}_{_type_token(dtype)}_"
        f"cc{cc}_ctx_{compile_context.symbol_suffix}"
    )
    if operation in {"rank", "count"}:
        cpp_type = _cpp_type(dtype)
        lines = [
            f'extern "C" __device__ {cpp_type} {stem}() {{',
            *_group_prelude(group),
            (
                f"  return static_cast<{cpp_type}>("
                f"{_query_expr(group, operation, level)});"
            ),
            "}",
        ]
        return_type = dtype
    elif operation == "is_member":
        stem = (
            f"cuda_coop_numba_mlir_group_{group.symbol_suffix}_is_member_"
            f"cc{cc}_ctx_{compile_context.symbol_suffix}"
        )
        lines = [
            f'extern "C" __device__ unsigned char {stem}() {{',
            *_group_prelude(group),
            "  return ::cuda::gpu_thread.is_part_of(group) ? 1u : 0u;",
            "}",
        ]
        return_type = types.uint8
    elif operation in {"sync", "sync_aligned"}:
        stem = (
            f"cuda_coop_numba_mlir_group_{group.symbol_suffix}_{operation}_"
            f"cc{cc}_ctx_{compile_context.symbol_suffix}"
        )
        member = "sync_aligned" if operation == "sync_aligned" else "sync"
        lines = [
            f'extern "C" __device__ void {stem}() {{',
            *_group_prelude(group),
            f"  group.{member}();",
            "}",
        ]
        return_type = types.void
    else:
        raise ValueError(f"unsupported group method operation {operation!r}")

    result = _RawCAbiInvocable(
        source=_source(lines),
        symbol=stem,
        return_type=return_type,
        cc=cc,
        compile_context=compile_context,
    )
    _GROUP_METHOD_INVOCABLE_CACHE[key] = result
    return result


__all__ = [
    "make_group_method_invocable",
]
