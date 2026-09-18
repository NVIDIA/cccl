# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block TopK specifications; the private CUB dependency is isolated here."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec, TypeDefinition
from .._bindings import ArgumentBinding, BindingKind
from .._types import (
    INT64,
    Array,
    CxxFunction,
    Dependency,
    TemplateParameter,
    TempStorageParameter,
    Value,
)
from ._common import normalize_block_dim, normalize_positive_int

# CUB has no public BlockTopK class. Keep method-template calls and their
# checked wide-integer boundary in one compatibility shim.
BLOCK_TOPK_TYPE = TypeDefinition(
    name="BlockTopKCoop",
    code=r"""
namespace cub {
template <typename KeyT, int BlockDimX, int ItemsPerThread, typename ValueT>
class BlockTopKCoop : public detail::block_topk<KeyT, BlockDimX, ItemsPerThread, ValueT>
{
  using base_t = detail::block_topk<KeyT, BlockDimX, ItemsPerThread, ValueT>;
  _CCCL_DEVICE_API _CCCL_FORCEINLINE bool validate_counts(long long k, long long num_valid)
  {
    constexpr long long tile_size = BlockDimX * ItemsPerThread;
    if (k < 0 || k > tile_size || num_valid < 0 || num_valid > tile_size)
    {
      asm volatile("trap;");
    }
    // CUB asserts that partial tiles are nonempty, even before its k check.
    return k != 0 && num_valid != 0;
  }
public:
  using base_t::base_t;
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void min_keys_full(KeyT (&keys)[ItemsPerThread],
    long long k, long long num_valid)
  {
    if (!validate_counts(k, num_valid))
    {
      return;
    }
    this->template min_keys<true>(keys, static_cast<int>(k), static_cast<int>(num_valid));
  }
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void min_keys_partial(KeyT (&keys)[ItemsPerThread],
    long long k, long long num_valid)
  {
    if (!validate_counts(k, num_valid))
    {
      return;
    }
    this->template min_keys<false>(keys, static_cast<int>(k), static_cast<int>(num_valid));
  }
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void min_pairs_full(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread],
    long long k, long long num_valid)
  {
    if (!validate_counts(k, num_valid))
    {
      return;
    }
    this->template min_pairs<true>(keys, values, static_cast<int>(k), static_cast<int>(num_valid));
  }
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void min_pairs_partial(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread],
    long long k, long long num_valid)
  {
    if (!validate_counts(k, num_valid))
    {
      return;
    }
    this->template min_pairs<false>(keys, values, static_cast<int>(k), static_cast<int>(num_valid));
  }
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void max_keys_full(KeyT (&keys)[ItemsPerThread],
    long long k, long long num_valid)
  {
    if (!validate_counts(k, num_valid))
    {
      return;
    }
    this->template max_keys<true>(keys, static_cast<int>(k), static_cast<int>(num_valid));
  }
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void max_keys_partial(KeyT (&keys)[ItemsPerThread],
    long long k, long long num_valid)
  {
    if (!validate_counts(k, num_valid))
    {
      return;
    }
    this->template max_keys<false>(keys, static_cast<int>(k), static_cast<int>(num_valid));
  }
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void max_pairs_full(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread],
    long long k, long long num_valid)
  {
    if (!validate_counts(k, num_valid))
    {
      return;
    }
    this->template max_pairs<true>(keys, values, static_cast<int>(k), static_cast<int>(num_valid));
  }
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void max_pairs_partial(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread],
    long long k, long long num_valid)
  {
    if (!validate_counts(k, num_valid))
    {
      return;
    }
    this->template max_pairs<false>(keys, values, static_cast<int>(k), static_cast<int>(num_valid));
  }
};
} // namespace cub
""".strip(),
)


def _count_parameter(binding: ArgumentBinding, *, name: str, tile_size: int):
    if binding.kind is BindingKind.RUNTIME:
        return Value(INT64, name=name)
    value = tile_size if binding.kind is BindingKind.OMITTED else binding.value
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"topk {name} must be an integer")
    if not 0 <= value <= tile_size:
        raise ValueError(f"topk {name} must be in [0, {tile_size}]")
    return CxxFunction(str(int(value)), INT64, name=name)


@dataclass(frozen=True)
class BlockTopKSpec:
    specialization: AlgorithmSpec
    block_dim: tuple[int, int, int]
    items_per_thread: int


def make_block_topk_spec(
    *,
    key_dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    selection: str,
    k: ArgumentBinding,
    num_valid: ArgumentBinding | None = None,
    value_dtype: Any | None = None,
) -> BlockTopKSpec:
    """Select an unsorted blocked prefix while preserving key/value association."""
    block_dim = normalize_block_dim(block_dim)
    if block_dim[1:] != (1, 1):
        raise ValueError("TopK supports only one-dimensional blocks")
    items_per_thread = normalize_positive_int("items_per_thread", items_per_thread)
    if selection not in {"min", "max"}:
        raise ValueError("topk selection must be min or max")
    if not isinstance(k, ArgumentBinding) or k.kind is BindingKind.OMITTED:
        raise TypeError("topk k requires an integer binding")
    num_valid = ArgumentBinding.omitted() if num_valid is None else num_valid
    if not isinstance(num_valid, ArgumentBinding):
        raise TypeError("topk num_valid must be an integer binding")
    tile_size = block_dim[0] * items_per_thread
    if tile_size > (1 << 31) - 1:
        raise ValueError("topk tile size must fit a signed 32-bit integer")
    parameters = [
        TempStorageParameter(),
        Array(
            Dependency("KeyT"),
            Dependency("ITEMS_PER_THREAD"),
            name="keys",
            is_inout=True,
        ),
    ]
    if value_dtype is not None:
        parameters.append(
            Array(
                Dependency("ValueT"),
                Dependency("ITEMS_PER_THREAD"),
                name="values",
                is_inout=True,
            )
        )
    parameters.extend(
        (
            _count_parameter(k, name="k", tile_size=tile_size),
            _count_parameter(num_valid, name="num_valid", tile_size=tile_size),
        )
    )
    payload = "keys" if value_dtype is None else "pairs"
    tile = "full" if num_valid.kind is BindingKind.OMITTED else "partial"
    algorithm = Algorithm(
        struct_name="BlockTopKCoop",
        method_name=f"{selection}_{payload}_{tile}",
        c_name="block_topk",
        includes=("cub/block/block_topk.cuh",),
        template_parameters=tuple(
            TemplateParameter(name)
            for name in ("KeyT", "BLOCK_DIM_X", "ITEMS_PER_THREAD", "ValueT")
        ),
        parameters=(tuple(parameters),),
        type_definitions=(BLOCK_TOPK_TYPE,),
    )
    spec = algorithm.specialize(
        {
            "KeyT": key_dtype,
            "BLOCK_DIM_X": block_dim[0],
            "ITEMS_PER_THREAD": items_per_thread,
            "ValueT": value_dtype if value_dtype is not None else "::cub::NullType",
        },
        metadata={
            "scope": "block",
            "primitive": "topk",
            "selection": selection,
            "payload": payload,
        },
    )
    return BlockTopKSpec(spec, block_dim, items_per_thread)
