# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402,F811

"""Reduction provider lowering for Numba-CUDA-MLIR.

Block, warp, and CUDAX group reduction providers share this semantic module.
Public markers and whole-function planning remain outside the provider layer.
"""

import operator
from typing import Any

import numpy as np
from numba_cuda_mlir import types

from cuda.coop._core import CxxOperator, Dependency, PythonOperator, ThreadGroup
from cuda.coop._core.block import make_block_reduce_spec
from cuda.coop._core.warp import make_warp_reduce_spec

from .._compiler import _nvrtc
from .._compiler._parameters import (
    CUB_BLOCK_REDUCE_ALGOS,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import make_invocable_from_specialization, numba_type_to_wrapper
from ._core import NumbaMlirCoreAdapter
from ._thread_group import (
    _cpp_type,
    _current_cc,
    _group_prelude,
    _RawCAbiInvocable,
    _source,
    _type_token,
)

_BUILTIN_REDUCE_OPERATORS = {
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}
_GROUP_REDUCE_INVOCABLE_CACHE: dict[tuple[Any, ...], _RawCAbiInvocable] = {}


def _positive_int(value, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _reduce_algorithm(algorithm) -> str:
    if isinstance(algorithm, bool):
        raise TypeError("block reduce algorithm must not be bool")
    if not isinstance(algorithm, str):
        raise TypeError("block reduce algorithm must be a string")
    if algorithm.startswith("::cub::BlockReduceAlgorithm::"):
        return algorithm
    try:
        return CUB_BLOCK_REDUCE_ALGOS[algorithm.lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(CUB_BLOCK_REDUCE_ALGOS))
        raise ValueError(
            f"Unsupported block reduce algorithm {algorithm!r}; "
            f"expected one of: {allowed}"
        ) from exc


def _materialize_block(
    *,
    dtype,
    threads_per_block,
    items_per_thread,
    operation,
    algorithm,
    reduce_operator=None,
    num_valid=None,
    methods=None,
):
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    if num_valid is not None and items_per_thread != 1:
        raise ValueError("num_valid is not supported for array inputs")
    dtype = normalize_dtype_param(dtype)
    core_spec = make_block_reduce_spec(
        dtype=dtype,
        block_dim=tuple(normalize_dim_param(threads_per_block)),
        items_per_thread=items_per_thread,
        operation=operation,
        algorithm=_reduce_algorithm(algorithm),
        value_kind="scalar" if items_per_thread == 1 else "array",
        reduce_operator=reduce_operator,
        valid_items=num_valid is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(specialization)


def sum(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="warp_reductions",
    num_valid=None,
    methods=None,
):
    """Build the direct CUB block-sum invocable selected by planning."""

    return _materialize_block(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        operation="sum",
        algorithm=algorithm,
        num_valid=num_valid,
        methods=methods,
    )


def reduce(
    dtype,
    threads_per_block=None,
    binary_op=None,
    items_per_thread=1,
    algorithm="warp_reductions",
    num_valid=None,
    methods=None,
):
    """Build a direct CUB block reduction with a device callback."""

    if not callable(binary_op):
        raise TypeError("binary_op must be a device callable")
    return _materialize_block(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        operation="reduce",
        algorithm=algorithm,
        reduce_operator=PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=binary_op,
            name="binary_op",
        ),
        num_valid=num_valid,
        methods=methods,
    )


def block_reduce_builtin(
    dtype,
    threads_per_block,
    binary_op,
    items_per_thread=1,
    algorithm="warp_reductions",
    num_valid=None,
):
    """Build a direct CUB reduction with a canonical C++ operator."""

    try:
        operator_cpp = _BUILTIN_REDUCE_OPERATORS[binary_op]
    except KeyError as exc:
        names = ", ".join(sorted(_BUILTIN_REDUCE_OPERATORS))
        raise ValueError(
            f"Unsupported built-in reduction {binary_op!r}; expected: {names}"
        ) from exc
    return _materialize_block(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        operation="reduce",
        algorithm=algorithm,
        reduce_operator=CxxOperator(
            cpp=operator_cpp,
            dtype=Dependency("T"),
            name="binary_op",
        ),
        num_valid=num_valid,
    )


def _materialize_warp(
    *,
    dtype,
    operation,
    threads_in_warp,
    valid_items,
    threads_per_block,
    reduce_operator=None,
    methods=None,
):
    dtype = normalize_dtype_param(dtype)
    threads_in_warp = _positive_int(threads_in_warp, name="threads_in_warp")
    core_spec = make_warp_reduce_spec(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
        operation=operation,
        reduce_operator=reduce_operator,
        valid_items=valid_items is not None,
        include_full_warp=valid_items is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(
        specialization,
        threads=threads_in_warp,
        block_threads=threads_per_block,
    )


def warp_sum(
    dtype,
    threads_in_warp=32,
    valid_items=None,
    threads_per_block=None,
):
    """Build the direct CUB warp-sum invocable selected by planning."""

    return _materialize_warp(
        dtype=dtype,
        operation="sum",
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        threads_per_block=threads_per_block,
    )


def warp_reduce(
    dtype,
    binary_op,
    threads_in_warp=32,
    valid_items=None,
    methods=None,
    threads_per_block=None,
):
    """Build a direct CUB warp reduction with a device callback."""

    if not callable(binary_op):
        raise TypeError("binary_op must be a device callable")
    return _materialize_warp(
        dtype=dtype,
        operation="reduce",
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        threads_per_block=threads_per_block,
        reduce_operator=PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=binary_op,
            name="binary_op",
        ),
        methods=methods,
    )


def warp_reduce_builtin(
    dtype,
    binary_op,
    threads_in_warp=32,
    valid_items=None,
    threads_per_block=None,
):
    """Build a direct CUB warp reduction with a canonical C++ operator."""

    try:
        operator_cpp = _BUILTIN_REDUCE_OPERATORS[binary_op]
    except KeyError as exc:
        names = ", ".join(sorted(_BUILTIN_REDUCE_OPERATORS))
        raise ValueError(
            f"Unsupported built-in reduction {binary_op!r}; expected: {names}"
        ) from exc
    return _materialize_warp(
        dtype=dtype,
        operation="reduce",
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        threads_per_block=threads_per_block,
        reduce_operator=CxxOperator(
            cpp=operator_cpp,
            dtype=Dependency("T"),
            name="binary_op",
        ),
    )


_CUDAX_REDUCE_OPS = {
    "sum": "::cuda::std::plus<>{}",
    "multiplies": "::cuda::std::multiplies<>{}",
    "min": "::cuda::minimum<>{}",
    "max": "::cuda::maximum<>{}",
    "bit_and": "::cuda::std::bit_and<>{}",
    "bit_or": "::cuda::std::bit_or<>{}",
    "bit_xor": "::cuda::std::bit_xor<>{}",
}
_CUDAX_REDUCE_OP_ALIASES = {
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
_CUDAX_CALLABLE_REDUCE_OP_ALIASES = {
    operator.add: "sum",
    operator.mul: "multiplies",
    operator.and_: "bit_and",
    operator.or_: "bit_or",
    operator.xor: "bit_xor",
    np.add: "sum",
    np.multiply: "multiplies",
    np.minimum: "min",
    np.maximum: "max",
    np.bitwise_and: "bit_and",
    np.bitwise_or: "bit_or",
    np.bitwise_xor: "bit_xor",
}


def _normalize_reduce_operation(binary_op: Any) -> str:
    """Map one public built-in reduction selector to its provider token."""

    try:
        return _CUDAX_REDUCE_OP_ALIASES[binary_op]
    except (KeyError, TypeError):
        pass
    try:
        return _CUDAX_CALLABLE_REDUCE_OP_ALIASES[binary_op]
    except (KeyError, TypeError):
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
    _compile_context: _nvrtc.CompileContext | None = None,
) -> _RawCAbiInvocable:
    """Compile the private CUDAX provider used by group-first lowering."""

    if not isinstance(group, ThreadGroup):
        raise TypeError("cuda.coop.numba_mlir.reduce group must be a ThreadGroup")
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
        compile_context=_compile_context,
    )


def make_group_reduce_invocable(
    *,
    group: ThreadGroup,
    dtype: Any,
    items_per_thread: int,
    operation: str,
    broadcast: bool,
    compile_context: _nvrtc.CompileContext | None = None,
) -> _RawCAbiInvocable:
    """Materialize a CUDAX group Reduce provider as inlinable LTO-IR."""

    if not isinstance(group, ThreadGroup):
        raise TypeError("cuda.coop.numba_mlir.reduce group must be a ThreadGroup")
    dtype = normalize_dtype_param(dtype)
    if group.kind == "grid":
        raise NotImplementedError(
            "cuda.coop.numba_mlir.reduce grid groups require a hidden "
            "per-launch workspace, which the Numba-CUDA-MLIR provider ABI "
            "does not expose yet"
        )
    if operation not in _CUDAX_REDUCE_OPS:
        allowed = ", ".join(sorted(_CUDAX_REDUCE_OPS))
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
    if not isinstance(broadcast, bool):
        raise TypeError("cuda.coop.numba_mlir.reduce broadcast must be a bool")

    cc = _current_cc()
    if compile_context is None:
        compile_context = _nvrtc.resolve_compile_context()
    key = (
        "reduce",
        group.semantic_key,
        dtype,
        items_per_thread,
        operation,
        broadcast,
        cc,
        compile_context,
    )
    cached = _GROUP_REDUCE_INVOCABLE_CACHE.get(key)
    if cached is not None:
        return cached

    cpp_type = _cpp_type(dtype)
    mode = "broadcast" if broadcast else "root"
    symbol = (
        "cuda_coop_numba_mlir_group_reduce_"
        f"{group.symbol_suffix}_{operation}_{_type_token(dtype)}_"
        f"x{items_per_thread}_{mode}_cc{cc}_ctx_{compile_context.symbol_suffix}"
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
    operator_cpp = _CUDAX_REDUCE_OPS[operation]
    if broadcast:
        lines.extend(
            [
                "  auto reduced = ::cuda::experimental::coop::reduce(",
                "      ::cuda::experimental::broadcasted, group, thread_data,",
                f"      {operator_cpp});",
                f"  {cpp_type} result = reduced;",
            ]
        )
    else:
        lines.extend(
            [
                "  auto reduced = ::cuda::experimental::coop::reduce(",
                f"      group, thread_data, {operator_cpp});",
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
        cc=cc,
        compile_context=compile_context,
    )
    _GROUP_REDUCE_INVOCABLE_CACHE[key] = result
    return result


__all__ = [
    "block_reduce_builtin",
    "group_reduce",
    "make_group_reduce_invocable",
    "reduce",
    "sum",
    "warp_reduce",
    "warp_reduce_builtin",
    "warp_sum",
]
