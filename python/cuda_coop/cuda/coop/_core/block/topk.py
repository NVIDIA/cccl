# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockTopK semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec, TypeDefinition
from .._bindings import ArgumentBinding, BindingKind, i32_parameter
from .._bindings import binding as binding
from .._types import (
    INT32,
    Array,
    Dependency,
    TemplateParameter,
    TempStorageParameter,
    Value,
)


class TopKSelection(str, Enum):
    MAX = "max"
    MIN = "min"


class TopKPayload(str, Enum):
    KEYS = "keys"
    PAIRS = "pairs"


class TopKTilePolicy(str, Enum):
    FULL = "full"
    PARTIAL = "partial"


_TEMPLATE_PARAMETERS = (
    TemplateParameter("KeyT"),
    TemplateParameter("BLOCK_DIM_X"),
    TemplateParameter("ITEMS_PER_THREAD"),
    TemplateParameter("ValueT"),
    TemplateParameter("BLOCK_DIM_Y"),
    TemplateParameter("BLOCK_DIM_Z"),
)


# CUB currently implements block TopK only as the undocumented
# ``cub::detail::block_topk``; there is no supported public ``cub::BlockTopK`` API.
# Keep that dependency isolated in this generated compatibility shim so every
# backend consumes the same C++ surface. The named forwarding methods also
# avoid making each backend encode C++ method-template syntax in its call and
# symbol machinery. This shim can go away if CUB gains a compatible public API.
BLOCK_TOPK_TYPE = TypeDefinition(
    name="BlockTopKCoop",
    code=r"""
namespace cub {
template <typename KeyT, int BlockDimX, int ItemsPerThread, typename ValueT, int BlockDimY, int BlockDimZ>
class BlockTopKCoop : public detail::block_topk<KeyT, BlockDimX, ItemsPerThread, ValueT>
{
  using base_t = detail::block_topk<KeyT, BlockDimX, ItemsPerThread, ValueT>;

public:
  // The core model uses the common X/Y/Z block specialization shape, while
  // detail::block_topk represents only BlockDimX. Reject dimensions it cannot
  // represent instead of silently discarding them.
  static_assert(BlockDimY == 1 && BlockDimZ == 1, "BlockTopKCoop only supports one-dimensional blocks");
  using base_t::base_t;

  // Turn the IsFullTile method-template argument into ordinary member names
  // that are stable call targets for all backend Algorithm implementations.
  __device__ __forceinline__ void max_keys_full(
    KeyT (&keys)[ItemsPerThread], int k, int num_valid,
    int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    this->template max_keys<true>(keys, k, num_valid, begin_bit, end_bit);
  }

  __device__ __forceinline__ void max_keys_partial(
    KeyT (&keys)[ItemsPerThread], int k, int num_valid,
    int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    this->template max_keys<false>(keys, k, num_valid, begin_bit, end_bit);
  }

  __device__ __forceinline__ void min_keys_full(
    KeyT (&keys)[ItemsPerThread], int k, int num_valid,
    int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    this->template min_keys<true>(keys, k, num_valid, begin_bit, end_bit);
  }

  __device__ __forceinline__ void min_keys_partial(
    KeyT (&keys)[ItemsPerThread], int k, int num_valid,
    int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    this->template min_keys<false>(keys, k, num_valid, begin_bit, end_bit);
  }

  __device__ __forceinline__ void max_pairs_full(
    KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread],
    int k, int num_valid, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    this->template max_pairs<true>(keys, values, k, num_valid, begin_bit, end_bit);
  }

  __device__ __forceinline__ void max_pairs_partial(
    KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread],
    int k, int num_valid, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    this->template max_pairs<false>(keys, values, k, num_valid, begin_bit, end_bit);
  }

  __device__ __forceinline__ void min_pairs_full(
    KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread],
    int k, int num_valid, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    this->template min_pairs<true>(keys, values, k, num_valid, begin_bit, end_bit);
  }

  __device__ __forceinline__ void min_pairs_partial(
    KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread],
    int k, int num_valid, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    this->template min_pairs<false>(keys, values, k, num_valid, begin_bit, end_bit);
  }
};
} // namespace cub
""".strip(),
)


@dataclass(frozen=True)
class BlockTopKSpec:
    """Fully specialized BlockTopK call semantics."""

    specialization: AlgorithmSpec
    selection: TopKSelection
    payload: TopKPayload
    tile_policy: TopKTilePolicy
    block_dim: tuple[int, int, int]
    items_per_thread: int
    num_valid: ArgumentBinding
    begin_bit: ArgumentBinding
    end_bit: ArgumentBinding

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def tile_size(self) -> int:
        x, y, z = self.block_dim
        return x * y * z * self.items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_block_topk_spec(
    *,
    key_dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    selection: str | TopKSelection,
    value_dtype: Any | None = None,
    num_valid: ArgumentBinding | None = None,
    begin_bit: ArgumentBinding | None = None,
    end_bit: ArgumentBinding | None = None,
) -> BlockTopKSpec:
    """Build canonical BlockTopK semantics from frontend-normalized inputs."""

    selection = TopKSelection(selection)
    block_dim = tuple(block_dim)
    if len(block_dim) != 3 or any(dim < 1 for dim in block_dim):
        raise ValueError("block_dim must contain three positive dimensions")
    if block_dim[1:] != (1, 1):
        raise ValueError("BlockTopK currently supports only 1D block dimensions")
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be positive")

    num_valid = ArgumentBinding.omitted() if num_valid is None else num_valid
    begin_bit = ArgumentBinding.omitted() if begin_bit is None else begin_bit
    end_bit = ArgumentBinding.omitted() if end_bit is None else end_bit
    if (begin_bit.kind is BindingKind.OMITTED) != (end_bit.kind is BindingKind.OMITTED):
        raise ValueError("begin_bit and end_bit must be provided together")

    payload = TopKPayload.PAIRS if value_dtype is not None else TopKPayload.KEYS
    tile_policy = (
        TopKTilePolicy.FULL
        if num_valid.kind is BindingKind.OMITTED
        else TopKTilePolicy.PARTIAL
    )
    method_name = f"{selection.value}_{payload.value}_{tile_policy.value}"
    tile_size = block_dim[0] * block_dim[1] * block_dim[2] * items_per_thread

    method: list[Any] = [
        TempStorageParameter(),
        Array(
            Dependency("KeyT"),
            Dependency("ITEMS_PER_THREAD"),
            name="keys",
            is_inout=True,
        ),
    ]
    if value_dtype is not None:
        method.append(
            Array(
                Dependency("ValueT"),
                Dependency("ITEMS_PER_THREAD"),
                name="values",
                is_inout=True,
            )
        )
    method.extend(
        (
            Value(INT32, name="k"),
            i32_parameter(num_valid, name="num_valid", omitted_value=tile_size),
        )
    )
    if begin_bit.kind is not BindingKind.OMITTED:
        method.extend(
            (
                i32_parameter(begin_bit, name="begin_bit"),
                i32_parameter(end_bit, name="end_bit"),
            )
        )

    algorithm = Algorithm(
        struct_name="BlockTopKCoop",
        method_name=method_name,
        c_name="block_topk",
        includes=("cub/block/block_topk.cuh",),
        template_parameters=_TEMPLATE_PARAMETERS,
        parameters=(tuple(method),),
        type_definitions=(BLOCK_TOPK_TYPE,),
    )
    specialization = algorithm.specialize(
        {
            "KeyT": key_dtype,
            "BLOCK_DIM_X": block_dim[0],
            "ITEMS_PER_THREAD": items_per_thread,
            "ValueT": value_dtype if value_dtype is not None else "::cub::NullType",
            "BLOCK_DIM_Y": block_dim[1],
            "BLOCK_DIM_Z": block_dim[2],
        },
        metadata={
            "scope": "block",
            "primitive": "topk",
            "selection": selection,
            "payload": payload,
            "tile_policy": tile_policy,
            "num_valid": num_valid.kind,
            "begin_bit": begin_bit.kind,
            "end_bit": end_bit.kind,
        },
    )
    return BlockTopKSpec(
        specialization=specialization,
        selection=selection,
        payload=payload,
        tile_policy=tile_policy,
        block_dim=block_dim,
        items_per_thread=items_per_thread,
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
