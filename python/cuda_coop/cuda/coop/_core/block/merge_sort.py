# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockMergeSort semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._symbols import semantic_token
from .._types import (
    INT32,
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
    KEYS = "keys"
    PAIRS = "pairs"


class BlockMergeSortTilePolicy(str, Enum):
    FULL = "full"
    PARTIAL = "partial"


_COMPARE_OPERATORS = (CxxOperator, PythonOperator)
_KEY_T = Dependency("KeyT")
_VALUE_T = Dependency("ValueT")
_ITEMS_PER_THREAD = Dependency("ITEMS_PER_THREAD")


@dataclass(frozen=True)
class BlockMergeSortSemantics:
    """Dimension-independent BlockMergeSort call contract.

    Runtime-width planning uses this contract directly. When a static block
    shape is known, :func:`make_block_merge_sort_spec` adds it.
    Runtime ``valid_items`` and ``oob_default`` payloads are intentionally not
    retained here. Their joint presence is recorded as the tile policy, which
    selects the CUB overload and participates in semantic identity.
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
class BlockMergeSortSpec:
    """Fully specialized CUB BlockMergeSort semantics."""

    specialization: AlgorithmSpec
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
    """Build the normalized BlockMergeSort call contract."""

    if key_dtype is None:
        raise ValueError("key dtype must be provided")
    if not isinstance(compare_operator, _COMPARE_OPERATORS):
        raise TypeError("BlockMergeSort requires a comparison operator")
    if isinstance(compare_operator, PythonOperator) and compare_operator.op is None:
        raise ValueError("compare_op must be provided")
    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread < 1
    ):
        raise ValueError("items_per_thread must be a positive integer")
    if (valid_items is None) != (oob_default is None):
        raise ValueError("valid_items and oob_default must be provided together")

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
                Value(INT32, name="valid_items"),
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


def make_block_merge_sort_spec(
    *,
    key_dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    compare_operator: CxxOperator | PythonOperator,
    value_dtype: Any | None = None,
    valid_items: Any = None,
    oob_default: Any = None,
) -> BlockMergeSortSpec:
    """Build a fully specialized CUB BlockMergeSort description."""

    block_dim = tuple(block_dim)
    if len(block_dim) != 3 or any(dim < 1 for dim in block_dim):
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
        struct_name="BlockMergeSort",
        method_name="Sort",
        c_name="block_merge_sort",
        includes=("cub/block/block_merge_sort.cuh",),
        template_parameters=(
            TemplateParameter("KeyT"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("ValueT"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ),
        parameters=(call.parameters,),
    ).specialize(
        {
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
    return BlockMergeSortSpec(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
    )
