# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Direct scalar CUB BlockReduce and WarpReduce lowering."""

from __future__ import annotations

import atexit
import functools
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.extending import overload, typing_registry
from numba_cuda_mlir.types import signature

from cuda.coop._core._bindings import ArgumentBinding
from cuda.coop._core.block.reduce import (
    BlockReduceAlgorithm,
    BlockReduceOperation,
    BlockReduceOperator,
    BlockReduceSpec,
    make_block_reduce_spec,
)
from cuda.coop._core.group import (
    GroupReduceSemantics,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.launch import LaunchFactOrigin, LaunchFacts
from cuda.coop._core.thread_group import this_block, this_warp
from cuda.coop._core.warp.reduce import WarpReduceSpec, make_warp_reduce_spec

from .._compiler import _nvrtc

_CPP_TYPES = {
    types.int8: "::cuda::std::int8_t",
    types.uint8: "::cuda::std::uint8_t",
    types.int16: "::cuda::std::int16_t",
    types.uint16: "::cuda::std::uint16_t",
    types.int32: "::cuda::std::int32_t",
    types.uint32: "::cuda::std::uint32_t",
    types.int64: "::cuda::std::int64_t",
    types.uint64: "::cuda::std::uint64_t",
    types.float32: "float",
    types.float64: "double",
}
_INTEGRAL_TYPES = frozenset(
    {
        types.int8,
        types.uint8,
        types.int16,
        types.uint16,
        types.int32,
        types.uint32,
        types.int64,
        types.uint64,
    }
)
_ALGORITHMS = {
    "raking_commutative_only": "::cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY",
    "raking": "::cub::BLOCK_REDUCE_RAKING",
    "warp_reductions": "::cub::BLOCK_REDUCE_WARP_REDUCTIONS",
}
_OPERATORS = {
    "multiplies": "::cuda::std::multiplies<T>{}",
    "min": "::cuda::minimum<T>{}",
    "max": "::cuda::maximum<T>{}",
    "bit_and": "::cuda::std::bit_and<T>{}",
    "bit_or": "::cuda::std::bit_or<T>{}",
    "bit_xor": "::cuda::std::bit_xor<T>{}",
}
_BITWISE_OPERATORS = frozenset({"bit_and", "bit_or", "bit_xor"})

_ARTIFACT_DIRECTORY = tempfile.TemporaryDirectory(prefix="cuda-coop-numba-mlir-")
atexit.register(_ARTIFACT_DIRECTORY.cleanup)


@dataclass(frozen=True)
class _Provider:
    extern: Any
    ltoir_path: str


@dataclass(frozen=True)
class _MarkerSpec:
    """Static reduction controls captured before payload typing."""

    group_kind: str
    block_dim: tuple[int, int, int]
    operation: BlockReduceOperation
    binary_op: BlockReduceOperator
    algorithm: BlockReduceAlgorithm | None
    valid_items: bool

    def __post_init__(self) -> None:
        if self.group_kind not in {"block", "warp"}:
            raise ValueError(f"unsupported reduction group kind {self.group_kind!r}")
        if self.group_kind == "warp" and self.algorithm is not None:
            raise ValueError("physical-warp reduction does not accept an algorithm")


def _unliteral(dtype: Any) -> Any:
    """Return the scalar type represented by a Numba literal type."""

    return getattr(dtype, "literal_type", dtype)


def _validate_scalar_dtype(dtype: Any, binary_op: str) -> Any:
    dtype = _unliteral(dtype)
    if dtype not in _CPP_TYPES:
        supported = ", ".join(str(item) for item in _CPP_TYPES)
        raise TypeError(
            f"cuda.coop group reduction does not support scalar type {dtype}; "
            f"expected one of: {supported}"
        )
    if binary_op in _BITWISE_OPERATORS and dtype not in _INTEGRAL_TYPES:
        raise TypeError(f"cuda.coop {binary_op} reduction requires an integer scalar")
    return dtype


def _block_source(*, symbol: str, cpp_type: str, spec: BlockReduceSpec) -> str:
    x, y, z = spec.block_dim
    binary_op = spec.binary_op.value
    method_args = "value"
    if spec.method_name == "Sum":
        method = "Sum"
    else:
        method = "Reduce"
        method_args += f", {_OPERATORS[binary_op]}"
    if spec.valid_items:
        method_args += ", valid_items"
    valid_parameter = ", int valid_items" if spec.valid_items else ""
    return f"""
#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

extern "C" __device__ {cpp_type} {symbol}({cpp_type} value{valid_parameter}) {{
  using T = {cpp_type};
  using BlockReduce = ::cub::BlockReduce<
    T, {x}, {_ALGORITHMS[spec.algorithm.value]}, {y}, {z}>;
  __shared__ typename BlockReduce::TempStorage storage;
  T result = BlockReduce(storage).{method}({method_args});
  __syncthreads();
  return result;
}}
"""


def _warp_source(*, symbol: str, cpp_type: str, spec: WarpReduceSpec) -> str:
    binary_op = spec.binary_op.value
    method_args = "value"
    if spec.method_name == "Sum":
        method = "Sum"
    else:
        method = "Reduce"
        method_args += f", {_OPERATORS[binary_op]}"
    if spec.valid_items:
        method_args += ", valid_items"
    valid_parameter = ", int valid_items" if spec.valid_items else ""
    return f"""
#include <cub/warp/warp_reduce.cuh>
#include <cuda/functional>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

extern "C" __device__ {cpp_type} {symbol}({cpp_type} value{valid_parameter}) {{
  using T = {cpp_type};
  using WarpReduce = ::cub::WarpReduce<T, {spec.threads_in_warp}>;
  __shared__ typename WarpReduce::TempStorage storage[{spec.warp_count}];
  const unsigned int linear_thread_rank =
    threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const unsigned int warp_id = linear_thread_rank / {spec.threads_in_warp};
  T result = WarpReduce(storage[warp_id]).{method}({method_args});
  __syncwarp();
  return result;
}}
"""


def _source(
    *,
    symbol: str,
    cpp_type: str,
    spec: BlockReduceSpec | WarpReduceSpec,
) -> str:
    """Render one direct CUB reduction provider."""

    if isinstance(spec, WarpReduceSpec):
        return _warp_source(symbol=symbol, cpp_type=cpp_type, spec=spec)
    return _block_source(symbol=symbol, cpp_type=cpp_type, spec=spec)


@functools.lru_cache(maxsize=None)
def _provider_for_context(
    spec: BlockReduceSpec | WarpReduceSpec,
    cpp_type: str,
    context: _nvrtc.CompileContext,
) -> _Provider:
    """Compile or reuse a provider for one exact target and toolchain."""

    scope = "warp" if isinstance(spec, WarpReduceSpec) else "block"
    source_template = _source(
        symbol=f"cuda_coop_{scope}_reduce_PROVIDER_DIGEST",
        cpp_type=cpp_type,
        spec=spec,
    )
    digest = hashlib.sha256(
        repr((source_template, spec.semantic_key, context)).encode()
    ).hexdigest()[:24]
    symbol = f"cuda_coop_{scope}_reduce_{digest}"
    source = _source(symbol=symbol, cpp_type=cpp_type, spec=spec)
    image = _nvrtc.compile_lto_ir(source, context)
    path = Path(_ARTIFACT_DIRECTORY.name, f"{symbol}.ltoir")
    path.write_bytes(image)
    parameters = [spec.dtype]
    if spec.valid_items:
        parameters.append(types.int32)
    extern = cuda.declare_device(
        symbol,
        signature(spec.dtype, *parameters),
        link=[str(path)],
        abi="c",
    )
    return _Provider(extern=extern, ltoir_path=str(path))


def _typed_provider(
    marker_spec: _MarkerSpec,
    dtype: Any,
    context: _nvrtc.CompileContext,
) -> _Provider:
    """Validate an overload payload type and select its direct-CUB provider."""

    dtype = _validate_scalar_dtype(dtype, marker_spec.binary_op.value)
    semantics = GroupReduceSemantics(
        dtype=dtype,
        operation=marker_spec.operation,
        binary_op=marker_spec.binary_op,
        algorithm=marker_spec.algorithm,
        valid_items=(
            ArgumentBinding.runtime()
            if marker_spec.valid_items
            else ArgumentBinding.omitted()
        ),
    )
    launch = LaunchFacts(
        exact_block_dim=marker_spec.block_dim,
        provenance=LaunchFactOrigin(
            fact="exact_block_dim",
            source="Numba-CUDA-MLIR typed provider",
            verified=True,
        ),
    )
    if marker_spec.group_kind == "block":
        group = this_block()
    elif marker_spec.group_kind == "warp":
        group = this_warp()
    else:
        raise ValueError(f"unsupported reduction group kind {marker_spec.group_kind!r}")
    plan = plan_group_primitive(
        make_group_primitive_call(
            group,
            semantics,
            source="numba-cuda-mlir overload typing",
        ),
        launch,
    ).require_supported()
    spec = plan.implementation
    assert spec is not None
    return _provider_for_context(spec, _CPP_TYPES[dtype], context)


@functools.lru_cache(maxsize=None)
def _marker_for(marker_spec: _MarkerSpec, context: _nvrtc.CompileContext):
    """Return a generic marker specialized by its overload payload type."""

    if marker_spec.valid_items:

        def marker(value, valid_items):
            del value, valid_items
            raise RuntimeError("cuda.coop reduction marker was not lowered")

        def typer(value, valid_items):
            if not isinstance(valid_items, types.Integer):
                raise TypeError("cuda.coop valid_items must be an integer")
            provider = _typed_provider(marker_spec, value, context)
            extern_fn = provider.extern

            def impl(value, valid_items):
                return extern_fn(value, valid_items)

            return impl

    else:

        def marker(value):
            del value
            raise RuntimeError("cuda.coop reduction marker was not lowered")

        def typer(value):
            provider = _typed_provider(marker_spec, value, context)
            extern_fn = provider.extern

            def impl(value):
                return extern_fn(value)

            return impl

    overload(marker, inline="always", typing_registry=typing_registry)(typer)
    return marker


def _materialize(
    *,
    threads_per_block: Any,
    operation: str,
    binary_op: Any,
    algorithm: Any,
    num_valid: bool,
    state: Any = None,
):
    normalized = make_block_reduce_spec(
        dtype=None,
        block_dim=threads_per_block,
        operation=operation,
        binary_op=binary_op,
        algorithm=algorithm,
        valid_items=num_valid,
    )
    marker_spec = _MarkerSpec(
        group_kind="block",
        block_dim=normalized.block_dim,
        operation=normalized.operation,
        binary_op=normalized.binary_op,
        algorithm=normalized.algorithm,
        valid_items=normalized.valid_items,
    )
    context = _nvrtc.resolve_compile_context(state)
    return _marker_for(marker_spec, context)


def _materialize_warp(
    *,
    threads_per_block: Any,
    operation: str,
    binary_op: Any,
    num_valid: bool,
    state: Any = None,
):
    normalized = make_warp_reduce_spec(
        dtype=None,
        block_dim=threads_per_block,
        operation=operation,
        binary_op=binary_op,
        valid_items=num_valid,
    )
    marker_spec = _MarkerSpec(
        group_kind="warp",
        block_dim=normalized.block_dim,
        operation=BlockReduceOperation(normalized.operation.value),
        binary_op=normalized.binary_op,
        algorithm=None,
        valid_items=normalized.valid_items,
    )
    context = _nvrtc.resolve_compile_context(state)
    return _marker_for(marker_spec, context)


def sum(
    threads_per_block: Any,
    algorithm: Any = "warp_reductions",
    num_valid: bool = False,
    *,
    _state: Any = None,
):
    """Materialize the block-sum factory selected by hierarchy planning."""

    return _materialize(
        threads_per_block=threads_per_block,
        operation="sum",
        binary_op="sum",
        algorithm=algorithm,
        num_valid=num_valid,
        state=_state,
    )


def block_reduce_builtin(
    threads_per_block: Any,
    binary_op: Any,
    algorithm: Any = "warp_reductions",
    num_valid: bool = False,
    *,
    _state: Any = None,
):
    """Materialize a direct built-in CUB BlockReduce factory."""

    return _materialize(
        threads_per_block=threads_per_block,
        operation="reduce",
        binary_op=binary_op,
        algorithm=algorithm,
        num_valid=num_valid,
        state=_state,
    )


def warp_sum(
    threads_per_block: Any,
    num_valid: bool = False,
    *,
    _state: Any = None,
):
    """Materialize the physical-warp sum selected by hierarchy planning."""

    return _materialize_warp(
        threads_per_block=threads_per_block,
        operation="sum",
        binary_op="sum",
        num_valid=num_valid,
        state=_state,
    )


def warp_reduce_builtin(
    threads_per_block: Any,
    binary_op: Any,
    num_valid: bool = False,
    *,
    _state: Any = None,
):
    """Materialize a direct built-in physical CUB WarpReduce factory."""

    return _materialize_warp(
        threads_per_block=threads_per_block,
        operation="reduce",
        binary_op=binary_op,
        num_valid=num_valid,
        state=_state,
    )


__all__ = ["block_reduce_builtin", "sum", "warp_reduce_builtin", "warp_sum"]
