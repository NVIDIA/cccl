# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Specialize scalar CUB BlockReduce calls for Numba-CUDA-MLIR."""

from __future__ import annotations

import atexit
import functools
import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.extending import overload, typing_registry
from numba_cuda_mlir.types import signature

from . import _nvrtc

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
class ReductionMarkerSpec:
    """Static specialization parameters carried by a dynamic marker."""

    block_dim: tuple[int, int, int]
    operation: str
    binary_op: str
    algorithm: str
    has_valid_items: bool


@dataclass(frozen=True)
class ProviderContext:
    """Resolved target and toolchain identity for one provider marker."""

    architecture: str
    nvrtc_version: tuple[int, int]
    include_paths: tuple[str, ...]


@dataclass(frozen=True)
class _Provider:
    extern: Any
    ltoir_path: str


def _unliteral(dtype: Any) -> Any:
    """Return the scalar type represented by a Numba literal type."""

    return getattr(dtype, "literal_type", dtype)


def _source(
    *,
    symbol: str,
    cpp_type: str,
    spec: ReductionMarkerSpec,
) -> str:
    x, y, z = spec.block_dim
    method_args = "value"
    if spec.operation == "sum" or spec.binary_op == "sum":
        method = "Sum"
    else:
        method = "Reduce"
        method_args += f", {_OPERATORS[spec.binary_op]}"
    if spec.has_valid_items:
        method_args += ", valid_items"
    valid_parameter = ", int valid_items" if spec.has_valid_items else ""
    return f"""
#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

extern "C" __device__ {cpp_type} {symbol}({cpp_type} value{valid_parameter}) {{
  using T = {cpp_type};
  using BlockReduce = ::cub::BlockReduce<
    T, {x}, {_ALGORITHMS[spec.algorithm]}, {y}, {z}>;
  __shared__ typename BlockReduce::TempStorage storage;
  T result = BlockReduce(storage).{method}({method_args});
  __syncthreads();
  return result;
}}
"""


def _target_architecture(state: Any = None) -> str:
    if state is not None:
        metadata = getattr(state, "metadata", None)
        targetoptions = (
            metadata.get("targetoptions") if isinstance(metadata, dict) else None
        )
        chip = targetoptions.get("chip") if isinstance(targetoptions, dict) else None
        if chip is not None:
            if not isinstance(chip, str):
                raise TypeError("Numba-CUDA-MLIR target chip must be a string")
            match = re.fullmatch(r"sm_([1-9][0-9]*(?:a|f)?)", chip)
            if match is None:
                raise ValueError(f"unsupported Numba-CUDA-MLIR target chip {chip!r}")
            return match.group(1)
    major, minor = cuda.get_current_device().compute_capability
    return f"{int(major)}{int(minor)}"


def resolve_provider_context(state: Any = None) -> ProviderContext:
    """Resolve context before choosing a cached overload marker."""

    return ProviderContext(
        architecture=_target_architecture(state),
        nvrtc_version=_nvrtc.version(),
        include_paths=_nvrtc.include_paths(),
    )


def _provider(
    dtype: Any,
    spec: ReductionMarkerSpec,
    context: ProviderContext | None = None,
) -> _Provider:
    dtype = _unliteral(dtype)
    try:
        cpp_type = _CPP_TYPES[dtype]
    except KeyError:
        supported = ", ".join(str(item) for item in _CPP_TYPES)
        raise TypeError(
            f"cuda.coop block reduction does not support scalar type {dtype}; "
            f"expected one of: {supported}"
        ) from None
    if spec.binary_op in _BITWISE_OPERATORS and dtype not in _INTEGRAL_TYPES:
        raise TypeError(
            f"cuda.coop {spec.binary_op} reduction requires an integer scalar"
        )

    context = context or resolve_provider_context()
    return _provider_for_context(dtype, spec, cpp_type, context)


@functools.lru_cache(maxsize=None)
def _provider_for_context(
    dtype: Any,
    spec: ReductionMarkerSpec,
    cpp_type: str,
    context: ProviderContext,
) -> _Provider:
    """Compile or reuse a provider for one resolved toolchain context."""

    source_template = _source(
        symbol="cuda_coop_block_reduce_PROVIDER_DIGEST",
        cpp_type=cpp_type,
        spec=spec,
    )
    digest_context = (
        source_template,
        str(dtype),
        context,
    )
    digest = hashlib.sha256(repr(digest_context).encode()).hexdigest()[:24]
    symbol = f"cuda_coop_block_reduce_{digest}"
    source = _source(symbol=symbol, cpp_type=cpp_type, spec=spec)
    image = _nvrtc.compile_lto_ir(
        source,
        context.architecture,
        context.include_paths,
    )
    path = Path(_ARTIFACT_DIRECTORY.name, f"{symbol}.ltoir")
    path.write_bytes(image)
    parameters = [dtype]
    if spec.has_valid_items:
        parameters.append(types.int32)
    extern = cuda.declare_device(
        symbol,
        signature(dtype, *parameters),
        link=[str(path)],
        abi="c",
    )
    return _Provider(extern=extern, ltoir_path=str(path))


@functools.lru_cache(maxsize=None)
def marker_for(spec: ReductionMarkerSpec, context: ProviderContext):
    """Return a marker whose overload infers and specializes scalar dtype."""

    if spec.has_valid_items:

        def marker(value, valid_items):
            del value, valid_items
            raise RuntimeError("cuda.coop reduction marker was not lowered")

        def typer(value, valid_items):
            dtype = _unliteral(value)
            if dtype not in _CPP_TYPES:
                raise TypeError(f"cuda.coop block reduction does not support {value}")
            if not isinstance(valid_items, types.Integer):
                raise TypeError("cuda.coop valid_items must be an integer")
            provider = _provider(dtype, spec, context)
            extern_fn = provider.extern
            ltoir_path = provider.ltoir_path

            def impl(value, valid_items):
                return extern_fn(value, valid_items)

            impl.__numba_cuda_mlir_link__ = [ltoir_path]
            return impl

    else:

        def marker(value):
            del value
            raise RuntimeError("cuda.coop reduction marker was not lowered")

        def typer(value):
            dtype = _unliteral(value)
            if dtype not in _CPP_TYPES:
                raise TypeError(f"cuda.coop block reduction does not support {value}")
            provider = _provider(dtype, spec, context)
            extern_fn = provider.extern
            ltoir_path = provider.ltoir_path

            def impl(value):
                return extern_fn(value)

            impl.__numba_cuda_mlir_link__ = [ltoir_path]
            return impl

    overload(marker, inline="always", typing_registry=typing_registry)(typer)
    return marker


__all__ = [
    "ProviderContext",
    "ReductionMarkerSpec",
    "marker_for",
    "resolve_provider_context",
]
