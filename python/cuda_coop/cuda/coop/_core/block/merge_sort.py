# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe CUB BlockMergeSort overloads without importing a compiler.

The call record stores two independent choices: keys only or key/value pairs,
and a full tile or a partial tile. Binding a block shape produces the
Algorithm consumed by a backend. CUB sorts its array operands in place; the
group API's separate results are implemented by the backend using copies.
Partial-tile wrappers check the wide runtime count before converting it to
CUB's integer argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, TypeDefinition
from .._symbols import semantic_token
from .._types import (
    INT64,
    Array,
    CxxOperator,
    Dependency,
    PythonOperator,
    Reference,
    TemplateParameter,
    TempStorageParameter,
    Value,
)


class BlockMergeSortPayload(str, Enum):
    """Select key-only sorting or keys with associated values."""

    KEYS = "keys"
    PAIRS = "pairs"


class BlockMergeSortTilePolicy(str, Enum):
    """Select a full tile or a valid prefix with a padding key."""

    FULL = "full"
    PARTIAL = "partial"


_COMPARE_OPERATORS = (CxxOperator, PythonOperator)
_KEY_T = Dependency("KeyT")
_VALUE_T = Dependency("ValueT")
_ITEMS_PER_THREAD = Dependency("ITEMS_PER_THREAD")


@dataclass(frozen=True)
class BlockMergeSortSemantics:
    """Describe a Merge Sort call before a block shape is known.

    Key and value dtypes, item count, comparator, and parameter order select
    the CUB call. The tile policy records whether both valid_items and
    oob_default are present. Their actual runtime values are not retained,
    so changing a count or padding key does not change this call identity.
    The same shape-independent record also feeds Warp group planning.
    """

    key_dtype: Any
    value_dtype: Any | None
    payload: BlockMergeSortPayload
    tile_policy: BlockMergeSortTilePolicy
    items_per_thread: int
    compare_operator: CxxOperator | PythonOperator
    parameters: tuple[Any, ...]

    @property
    def has_values(self) -> bool:
        return self.payload is BlockMergeSortPayload.PAIRS

    @property
    def has_partial_tile(self) -> bool:
        return self.tile_policy is BlockMergeSortTilePolicy.PARTIAL

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "block_merge_sort",
            semantic_token(self.key_dtype),
            semantic_token(self.value_dtype),
            self.payload.value,
            self.tile_policy.value,
            self.items_per_thread,
            semantic_token(self.compare_operator),
            semantic_token(self.parameters),
        )


@dataclass(frozen=True)
class BlockMergeSortSpecialization:
    """Pair a bound CUB Algorithm with its block shape and call choices.

    Backends use the Algorithm to emit the provider. The retained call record
    exposes payload and tile policy without inspecting its C++ parameters.
    """

    specialization: Algorithm
    call: BlockMergeSortSemantics
    block_dim: tuple[int, int, int]

    @property
    def key_dtype(self) -> Any:
        return self.call.key_dtype

    @property
    def value_dtype(self) -> Any | None:
        return self.call.value_dtype

    @property
    def payload(self) -> BlockMergeSortPayload:
        return self.call.payload

    @property
    def tile_policy(self) -> BlockMergeSortTilePolicy:
        return self.call.tile_policy

    @property
    def items_per_thread(self) -> int:
        return self.call.items_per_thread

    @property
    def compare_operator(self) -> CxxOperator | PythonOperator:
        return self.call.compare_operator

    @property
    def has_values(self) -> bool:
        return self.call.has_values

    @property
    def has_partial_tile(self) -> bool:
        return self.call.has_partial_tile

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_block_merge_sort_semantics(
    *,
    key_dtype: Any,
    items_per_thread: int,
    compare_operator: CxxOperator | PythonOperator,
    value_dtype: Any | None = None,
    valid_items: Any = None,
    oob_default: Any = None,
) -> BlockMergeSortSemantics:
    """Validate payload choices and build the CUB parameter sequence.

    Keys and optional values are inout arrays. A partial tile adds a wide
    count and a key-typed padding value after the comparator. This function
    checks only their joint presence; it does not inspect device values.

    Parameters
    ----------
    key_dtype : object
        Required key type understood by the consuming backend.
    items_per_thread : int
        Positive fixed array extent, excluding booleans.
    compare_operator : CxxOperator or PythonOperator
        Static comparison descriptor. A Python descriptor must identify a
        callable; the backend validates and compiles its type contract.
    value_dtype : object, optional
        Associated-value type. Omit it for a keys-only overload.
    valid_items, oob_default : object, optional
        Supply both to select the partial-tile overload. Their values stay
        runtime operands and are not embedded in this record.

    Returns
    -------
    BlockMergeSortSemantics
        Payload and tile choices with ordered parameter descriptors.

    Raises
    ------
    TypeError
        The comparator is not a supported operator descriptor.
    ValueError
        The key type or Python callable is missing, the extent is invalid,
        or only one partial-tile control is supplied.
    """

    if key_dtype is None:
        raise ValueError("key dtype must be provided")
    if not isinstance(compare_operator, _COMPARE_OPERATORS):
        raise TypeError("BlockMergeSort requires a comparison operator")
    if (
        isinstance(compare_operator, PythonOperator)
        and compare_operator.op is None
    ):
        raise ValueError("compare_op must be provided")
    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread < 1
    ):
        raise ValueError("items_per_thread must be a positive integer")
    if (valid_items is None) != (oob_default is None):
        raise ValueError(
            "valid_items and oob_default must be provided together"
        )

    payload = (
        BlockMergeSortPayload.PAIRS
        if value_dtype is not None
        else BlockMergeSortPayload.KEYS
    )
    tile_policy = (
        BlockMergeSortTilePolicy.PARTIAL
        if valid_items is not None
        else BlockMergeSortTilePolicy.FULL
    )
    parameters: list[Any] = [
        TempStorageParameter(),
        Array(
            _KEY_T,
            _ITEMS_PER_THREAD,
            name="keys",
            is_inout=True,
            is_return=False,
        ),
    ]
    if payload is BlockMergeSortPayload.PAIRS:
        parameters.append(
            Array(
                _VALUE_T,
                _ITEMS_PER_THREAD,
                name="values",
                is_inout=True,
                is_return=False,
            )
        )
    parameters.append(compare_operator)
    if tile_policy is BlockMergeSortTilePolicy.PARTIAL:
        parameters.extend(
            (
                Value(INT64, name="valid_items"),
                Reference(_KEY_T, name="oob_default"),
            )
        )

    return BlockMergeSortSemantics(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        payload=payload,
        tile_policy=tile_policy,
        items_per_thread=items_per_thread,
        compare_operator=compare_operator,
        parameters=tuple(parameters),
    )


def make_block_merge_sort_specialization(
    *,
    key_dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    compare_operator: CxxOperator | PythonOperator,
    value_dtype: Any | None = None,
    valid_items: Any = None,
    oob_default: Any = None,
) -> BlockMergeSortSpecialization:
    """Bind a block shape to the full or checked partial CUB Sort overload.

    The partial form uses a generated adapter around CUB. It accepts the
    count as a signed 64-bit value, checks zero through the group capacity,
    and only then narrows it. Full tiles call the CUB primitive directly.

    Parameters
    ----------
    key_dtype : object
        Key type passed to make_block_merge_sort_semantics.
    block_dim : tuple of int
        Three positive Python dimensions, excluding booleans. Their product
        must be a power of two for cub::BlockMergeSort.
    items_per_thread : int
        Positive array extent shared by the participating threads.
    compare_operator : CxxOperator or PythonOperator
        Comparison descriptor included in the method parameters.
    value_dtype : object, optional
        Associated-value type. Omit it for keys-only sorting.
    valid_items, oob_default : object, optional
        Joint presence selects partial-tile parameters. The backend supplies
        their runtime operands; the factory does not retain their values.

    Returns
    -------
    BlockMergeSortSpecialization
        Bound Algorithm plus the call record and exact block dimensions.
        Its input arrays are also the outputs; it has no scalar return.

    Raises
    ------
    TypeError
        The comparison descriptor is unsupported.
    ValueError
        The block shape or shared call arguments violate the constraints.
    """

    block_dim = tuple(block_dim)
    if len(block_dim) != 3 or any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim < 1
        for dim in block_dim
    ):
        raise ValueError("block_dim must contain three positive dimensions")
    block_threads = block_dim[0] * block_dim[1] * block_dim[2]
    if block_threads & (block_threads - 1):
        raise ValueError(
            "cub::BlockMergeSort requires a power-of-two block thread count"
        )
    call = make_block_merge_sort_semantics(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        items_per_thread=items_per_thread,
        compare_operator=compare_operator,
        valid_items=valid_items,
        oob_default=oob_default,
    )
    specialization = Algorithm(
        struct_name="CudaCoopBlockMergeSort"
        if call.has_partial_tile
        else "BlockMergeSort",
        method_name="Sort",
        c_name="block_merge_sort",
        includes=("cub/block/block_merge_sort.cuh", "cuda/std/functional"),
        type_definitions=(
            (_CHECKED_MERGE_SORT, _CHECKED_BLOCK_MERGE_SORT)
            if call.has_partial_tile
            else ()
        ),
        template_parameters=(
            TemplateParameter("KeyT"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("ValueT"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ),
        parameters=(call.parameters,),
        template_arguments={
            "KeyT": key_dtype,
            "BLOCK_DIM_X": block_dim[0],
            "ITEMS_PER_THREAD": items_per_thread,
            "ValueT": value_dtype if call.has_values else "::cub::NullType",
            "BLOCK_DIM_Y": block_dim[1],
            "BLOCK_DIM_Z": block_dim[2],
        },
        metadata={
            "scope": "block",
            "primitive": "merge_sort",
            "payload": call.payload,
            "tile_policy": call.tile_policy,
            "operator": type(compare_operator).__qualname__,
        },
    )
    return BlockMergeSortSpecialization(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
    )


_CHECKED_MERGE_SORT = TypeDefinition(
    name="cuda_coop_checked_merge_sort",
    code=r"""
namespace cub {
template <typename PrimitiveT, typename KeyT, typename ValueT,
          int ItemsPerThread, int TileSize>
struct CudaCoopCheckedMergeSort : PrimitiveT
{
  using PrimitiveT::PrimitiveT;

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Sort(
    KeyT (&keys)[ItemsPerThread], CompareOp compare_op,
    long long valid_items, KeyT oob_default)
  {
    if (valid_items < 0 || valid_items > TileSize)
    {
      asm volatile("trap;");
    }
    PrimitiveT::Sort(keys, compare_op, static_cast<int>(valid_items), oob_default);
  }

  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Sort(
    KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread],
    CompareOp compare_op, long long valid_items, KeyT oob_default)
  {
    if (valid_items < 0 || valid_items > TileSize)
    {
      asm volatile("trap;");
    }
    PrimitiveT::Sort(keys, values, compare_op,
                    static_cast<int>(valid_items), oob_default);
  }
};
}
""",  # noqa: E501 - Embedded C++ source.
)

_CHECKED_BLOCK_MERGE_SORT = TypeDefinition(
    name="cuda_coop_checked_block_merge_sort",
    code=r"""
namespace cub {
template <typename KeyT, int BlockDimX, int ItemsPerThread, typename ValueT,
          int BlockDimY, int BlockDimZ>
using CudaCoopBlockMergeSort = CudaCoopCheckedMergeSort<
  BlockMergeSort<KeyT, BlockDimX, ItemsPerThread, ValueT, BlockDimY, BlockDimZ>,
  KeyT, ValueT, ItemsPerThread, BlockDimX * BlockDimY * BlockDimZ * ItemsPerThread>;
}
""",  # noqa: E501 - Embedded C++ source.
)
