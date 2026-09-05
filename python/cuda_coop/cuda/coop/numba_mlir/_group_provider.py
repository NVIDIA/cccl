# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LTO-IR providers for Numba-CUDA-MLIR CUDAX group operations."""

from __future__ import annotations

import hashlib
import operator
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

from . import _nvrtc
from ._common import make_binary_tempfile, normalize_dtype_param
from ._types import (
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
_LTOIR_CACHE: dict[tuple[int, str], bytes] = {}
_INVOCABLE_CACHE: dict[tuple[Any, ...], "_RawCAbiInvocable"] = {}


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


def _compile_ltoir(source: str) -> bytes:
    cc = _current_cc()
    key = (cc, hashlib.sha1(source.encode("utf-8")).hexdigest())
    cached = _LTOIR_CACHE.get(key)
    if cached is not None:
        return cached
    _, image = _nvrtc.compile(cpp=source, cc=cc, rdc=True, code="lto")
    result = bytes(image)
    _LTOIR_CACHE[key] = result
    return result


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
        expected_types: tuple[Any, ...] = (),
        abi_types: tuple[Any, ...] = (),
        transforms: tuple[str, ...] = (),
    ) -> None:
        if not (len(expected_types) == len(abi_types) == len(transforms)):
            raise ValueError("group provider ABI metadata has inconsistent arity")

        self._temp_file = make_binary_tempfile(_compile_ltoir(source), ".ltoir")
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
    return [
        *render_hierarchy_decl(group.hierarchy),
        *render_group_decl_lines(group),
    ]


def _query_expr(group: ThreadGroup, operation: str, level: str) -> str:
    level_expr = cpp_level_expr(level)
    group_level = group.kind if group.mapping is None else group.mapping.parent
    if group.mapping is None and level == group_level:
        return "0" if operation == "rank" else "1"
    if _LEVEL_ORDER[level] < _LEVEL_ORDER[group_level]:
        return f"{level_expr}.{operation}(group)"
    if group.mapping is not None:
        # Mapped cudax groups answer only parent-relative queries; finer
        # levels observe the group through the level object above.
        return f"group.{operation}(group_parent)"
    return f"group.{operation}({level_expr})"


def make_group_method_invocable(
    *,
    group: ThreadGroup,
    operation: str,
    dtype: Any = None,
    level: str = "thread",
) -> _RawCAbiInvocable:
    """Materialize a query, membership, or synchronization helper."""

    dtype = types.int32 if dtype is None else normalize_dtype_param(dtype)
    if operation in {"rank", "count"}:
        validate_thread_group_query_dtype(dtype, scope="cuda.coop.numba_mlir")
    key = ("method", group.semantic_key, operation, dtype, level, _current_cc())
    cached = _INVOCABLE_CACHE.get(key)
    if cached is not None:
        return cached

    stem = (
        "cuda_coop_numba_mlir_group_"
        f"{group.symbol_suffix}_{operation}_{level}_{_type_token(dtype)}"
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
        stem = f"cuda_coop_numba_mlir_group_{group.symbol_suffix}_is_member"
        lines = [
            f'extern "C" __device__ unsigned char {stem}() {{',
            *_group_prelude(group),
            "  return ::cuda::gpu_thread.is_part_of(group) ? 1u : 0u;",
            "}",
        ]
        return_type = types.uint8
    elif operation in {"sync", "sync_aligned"}:
        stem = f"cuda_coop_numba_mlir_group_{group.symbol_suffix}_{operation}"
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
    )
    _INVOCABLE_CACHE[key] = result
    return result


_REDUCE_OPS = {
    "sum": "::cuda::std::plus<>{}",
    "multiplies": "::cuda::std::multiplies<>{}",
    "min": "::cuda::minimum<>{}",
    "max": "::cuda::maximum<>{}",
    "bit_and": "::cuda::std::bit_and<>{}",
    "bit_or": "::cuda::std::bit_or<>{}",
    "bit_xor": "::cuda::std::bit_xor<>{}",
}
_REDUCE_OP_ALIASES = {
    None: "sum",
    "+": "sum",
    "sum": "sum",
    "add": "sum",
    "plus": "sum",
    "*": "multiplies",
    "mul": "multiplies",
    "multiply": "multiplies",
    "multiplies": "multiplies",
    "min": "min",
    "minimum": "min",
    "max": "max",
    "maximum": "max",
    "&": "bit_and",
    "bit_and": "bit_and",
    "|": "bit_or",
    "bit_or": "bit_or",
    "^": "bit_xor",
    "bit_xor": "bit_xor",
}
_CALLABLE_REDUCE_OP_ALIASES = {
    operator.add: "sum",
    operator.mul: "multiplies",
    operator.and_: "bit_and",
    operator.or_: "bit_or",
    operator.xor: "bit_xor",
}
_CALLABLE_REDUCE_OP_NAME_ALIASES = {
    ("_operator", "add"): "sum",
    ("_operator", "mul"): "multiplies",
    ("_operator", "and_"): "bit_and",
    ("_operator", "or_"): "bit_or",
    ("_operator", "xor"): "bit_xor",
    ("operator", "add"): "sum",
    ("operator", "mul"): "multiplies",
    ("operator", "and_"): "bit_and",
    ("operator", "or_"): "bit_or",
    ("operator", "xor"): "bit_xor",
    ("numpy", "add"): "sum",
    ("numpy", "multiply"): "multiplies",
    ("numpy", "minimum"): "min",
    ("numpy", "maximum"): "max",
    ("numpy", "bitwise_and"): "bit_and",
    ("numpy", "bitwise_or"): "bit_or",
    ("numpy", "bitwise_xor"): "bit_xor",
}


def _normalize_reduce_operation(binary_op: Any) -> str:
    try:
        return _REDUCE_OP_ALIASES[binary_op]
    except (KeyError, TypeError):
        pass
    try:
        return _CALLABLE_REDUCE_OP_ALIASES[binary_op]
    except (KeyError, TypeError):
        pass
    if callable(binary_op):
        key = (
            getattr(binary_op, "__module__", ""),
            getattr(binary_op, "__name__", ""),
        )
        try:
            return _CALLABLE_REDUCE_OP_NAME_ALIASES[key]
        except KeyError:
            pass
    raise NotImplementedError(
        "cuda.coop.numba_mlir.reduce currently supports sum, multiplies, min, "
        "max, bit_and, bit_or, and bit_xor reductions"
    )


def group_reduce(
    dtype: Any,
    group: ThreadGroup,
    binary_op: Any = None,
    items_per_thread: int = 1,
    broadcast: bool = True,
    methods: Any = None,
) -> _RawCAbiInvocable:
    """Compile the private provider used by the group-first rewrite."""

    _validate_group_reduce_support(group)
    if not isinstance(broadcast, bool):
        raise TypeError("cuda.coop.numba_mlir.reduce broadcast must be a bool")
    if methods is not None:
        raise NotImplementedError(
            "cuda.coop.numba_mlir.reduce CUDAX lowering does not yet support "
            "custom dtype methods"
        )
    return make_group_reduce_invocable(
        group=group,
        dtype=dtype,
        items_per_thread=items_per_thread,
        operation=_normalize_reduce_operation(binary_op),
        broadcast=broadcast,
    )


def _validate_group_reduce_support(group: ThreadGroup) -> None:
    """Reject group kinds whose current lowering has incorrect semantics."""

    if not isinstance(group, ThreadGroup):
        raise TypeError("cuda.coop.numba_mlir.reduce group must be a ThreadGroup")
    if group.kind == "warps_within_block":
        raise NotImplementedError(
            "cuda.coop.numba_mlir reduce/sum does not support "
            "warps_within_block groups because the current CUDAX mapping "
            "does not preserve independent mapped-group reduction semantics"
        )


def make_group_reduce_invocable(
    *,
    group: ThreadGroup,
    dtype: Any,
    items_per_thread: int,
    operation: str,
    broadcast: bool,
) -> _RawCAbiInvocable:
    """Materialize a CUDAX group Reduce provider without a late PTX boundary."""

    _validate_group_reduce_support(group)
    dtype = normalize_dtype_param(dtype)
    if group.kind == "grid":
        raise NotImplementedError(
            "cuda.coop.numba_mlir.reduce grid groups require a hidden "
            "per-launch workspace, which the Numba-CUDA-MLIR provider ABI "
            "does not expose yet"
        )
    if operation not in _REDUCE_OPS:
        allowed = ", ".join(sorted(_REDUCE_OPS))
        raise NotImplementedError(
            "cuda.coop.numba_mlir.reduce CUDAX lowering supports built-in "
            f"operators {{{allowed}}}; got {operation!r}"
        )
    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread < 1
    ):
        raise ValueError("items_per_thread must be a positive integer")

    key = (
        "reduce",
        group.semantic_key,
        dtype,
        items_per_thread,
        operation,
        broadcast,
        _current_cc(),
    )
    cached = _INVOCABLE_CACHE.get(key)
    if cached is not None:
        return cached

    cpp_type = _cpp_type(dtype)
    mode = "broadcast" if broadcast else "root"
    symbol = (
        "cuda_coop_numba_mlir_group_reduce_"
        f"{group.symbol_suffix}_{operation}_{_type_token(dtype)}_"
        f"x{items_per_thread}_{mode}"
    )
    if items_per_thread == 1:
        parameter = f"{cpp_type} item"
        expected_types = (dtype,)
        abi_types = (dtype,)
        transforms = ("value",)
        thread_data = f"  {cpp_type} thread_data[1] = {{item}};"
    else:
        parameter = "void* raw_items"
        expected_types = (types.Array(dtype, 1, "C"),)
        abi_types = (types.CPointer(types.none),)
        transforms = ("ptr",)
        thread_data = (
            f"  auto& thread_data = *reinterpret_cast<{cpp_type} "
            f"(*)[{items_per_thread}]>(raw_items);"
        )

    lines = [
        f'extern "C" __device__ {cpp_type} {symbol}({parameter}) {{',
        *_group_prelude(group),
    ]
    if group.mapping is not None and group.complete_membership is False:
        lines.extend(
            [
                "  if (!::cuda::gpu_thread.is_part_of(group)) {",
                f"    return {cpp_type}{{}};",
                "  }",
            ]
        )
    lines.append(thread_data)
    operator = _REDUCE_OPS[operation]
    if broadcast:
        lines.extend(
            [
                "  auto reduced = ::cuda::experimental::coop::reduce(",
                "      ::cuda::experimental::broadcasted, group, thread_data,",
                f"      {operator});",
                f"  {cpp_type} result = reduced;",
            ]
        )
    else:
        lines.extend(
            [
                "  auto reduced = ::cuda::experimental::coop::reduce(",
                f"      group, thread_data, {operator});",
                f"  {cpp_type} result = reduced.value_or({cpp_type}{{}});",
            ]
        )
    if group.kind != "thread":
        lines.append("  group.sync_aligned();")
    lines.extend(("  return result;", "}"))

    result = _RawCAbiInvocable(
        source=_source(lines),
        symbol=symbol,
        return_type=dtype,
        expected_types=expected_types,
        abi_types=abi_types,
        transforms=transforms,
    )
    _INVOCABLE_CACHE[key] = result
    return result


__all__ = [
    "group_reduce",
    "make_group_method_invocable",
    "make_group_reduce_invocable",
]
