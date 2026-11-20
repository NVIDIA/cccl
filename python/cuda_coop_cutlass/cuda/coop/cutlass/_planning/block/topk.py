# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build compiler-neutral specializations of CUB's private block TopK.

A compatibility shim gives the method-template calls fixed entry points.
It checks wide count arguments before narrowing them to CUB's integer type.
Zero counts skip the native call. This avoids CUB's nonempty partial-tile
requirement even when no items are requested.
The native call modifies its payloads; public APIs must provide working copies
to preserve their inputs. Constructing a specialization does not run a kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .._algorithm import Algorithm, TypeDefinition
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
""".strip(),  # noqa: E501 - Embedded C++ source.
)


def _count_parameter(binding: ArgumentBinding, *, name: str, tile_size: int):
    """Represent a count as an int64 runtime input or an inline constant.

    Omission means the full tile size. Validate known values here; the shim
    checks runtime values before narrowing them to CUB's int count type.
    """

    if binding.kind is BindingKind.RUNTIME:
        return Value(INT64, name=name)
    value = tile_size if binding.kind is BindingKind.OMITTED else binding.value
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"topk {name} must be an integer")
    if not 0 <= value <= tile_size:
        raise ValueError(f"topk {name} must be in [0, {tile_size}]")
    return CxxFunction(str(int(value)), INT64, name=name)


@dataclass(frozen=True)
class BlockTopKSpecialization:
    """Keep a specialized TopK algorithm with its normalized payload shape.

    Attributes
    ----------
    specialization : Algorithm
        In-place CUB key or pair operation and its scratch/count parameters.
    block_dim : tuple of int
        One-dimensional block shape ``(threads, 1, 1)``.
    items_per_thread : int
        Fixed number of keys, and optional values, contributed by each thread.
    """

    specialization: Algorithm
    block_dim: tuple[int, int, int]
    items_per_thread: int


def make_block_topk_specialization(
    *,
    key_dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    selection: str,
    k: ArgumentBinding,
    num_valid: ArgumentBinding | None = None,
    value_dtype: Any | None = None,
) -> BlockTopKSpecialization:
    """Build a TopK specialization for a fixed one-dimensional block.

    Bind dtypes, per-thread extent, selection direction, and count policies.
    The resulting native call modifies the supplied arrays in place. Compiler
    adapters preserve public inputs by using separate working copies.

    Parameters
    ----------
    key_dtype : object
        Compiler-neutral key type used for the CUB template argument.
    block_dim : tuple of int
        Positive launch dimensions; the second and third dimensions must be 1.
    items_per_thread : int
        Positive, fixed key count per thread. The full tile size must fit
        a signed 32-bit integer.
    selection : {"min", "max"}
        Whether to select the smallest or largest keys.
    k : ArgumentBinding
        Required static or runtime count. Known counts must be between zero
        and the tile size; the shim checks runtime counts before narrowing.
    num_valid : ArgumentBinding or None, optional
        Valid input count. None or an omitted binding selects the full-tile
        method. A static or runtime binding selects the partial-tile method,
        even when its value equals the tile size.
    value_dtype : object or None, optional
        Associated value type. Omit it for a keys-only specialization.

    Returns
    -------
    BlockTopKSpecialization
        Specialized Algorithm and normalized block/payload shape. Only the
        first ``min(k, num_valid)`` blocked positions are defined by the call.
        Selection does not order that prefix; pairs retain their association.
    """

    block_dim = normalize_block_dim(block_dim)
    if block_dim[1:] != (1, 1):
        raise ValueError("TopK supports only one-dimensional blocks")
    items_per_thread = normalize_positive_int(
        "items_per_thread", items_per_thread
    )
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
    specialization = Algorithm(
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
        template_arguments={
            "KeyT": key_dtype,
            "BLOCK_DIM_X": block_dim[0],
            "ITEMS_PER_THREAD": items_per_thread,
            "ValueT": value_dtype
            if value_dtype is not None
            else "::cub::NullType",
        },
        metadata={
            "scope": "block",
            "primitive": "topk",
            "selection": selection,
            "payload": payload,
        },
    )
    return BlockTopKSpecialization(specialization, block_dim, items_per_thread)
