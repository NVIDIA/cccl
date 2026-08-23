# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB WarpMergeSort semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
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
from ._common import _validate_items_per_thread, _validate_logical_warp_threads


class WarpMergeSortPayload(str, Enum):
    KEYS = "keys"
    PAIRS = "pairs"


class WarpMergeSortTilePolicy(str, Enum):
    FULL = "full"
    PARTIAL = "partial"


_COMPARE_OPERATORS = (CxxOperator, PythonOperator)
_KEY_T = Dependency("KeyT")
_VALUE_T = Dependency("ValueT")
_ITEMS_PER_THREAD = Dependency("ITEMS_PER_THREAD")


@dataclass(frozen=True)
class WarpMergeSortSpec:
    """Fully specialized WarpMergeSort call semantics."""

    specialization: AlgorithmSpec
    payload: WarpMergeSortPayload
    tile_policy: WarpMergeSortTilePolicy
    key_dtype: Any
    value_dtype: Any | None
    items_per_thread: int
    threads_in_warp: int
    compare_operator: CxxOperator | PythonOperator = field(compare=False, hash=False)

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def has_values(self) -> bool:
        return self.payload is WarpMergeSortPayload.PAIRS

    @property
    def has_partial_tile(self) -> bool:
        return self.tile_policy is WarpMergeSortTilePolicy.PARTIAL

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_warp_merge_sort_spec(
    *,
    key_dtype: Any,
    items_per_thread: int,
    threads_in_warp: int,
    compare_operator: CxxOperator | PythonOperator,
    value_dtype: Any | None = None,
    valid_items: Any = None,
    oob_default: Any = None,
) -> WarpMergeSortSpec:
    """Build canonical WarpMergeSort keys or pairs semantics."""

    if key_dtype is None:
        raise ValueError("key dtype must be provided")
    if not isinstance(compare_operator, _COMPARE_OPERATORS):
        raise TypeError("WarpMergeSort requires a comparison operator")
    if isinstance(compare_operator, PythonOperator) and compare_operator.op is None:
        raise ValueError("compare_op must be provided")
    if (valid_items is None) != (oob_default is None):
        raise ValueError("valid_items and oob_default must be provided together")

    items_per_thread = _validate_items_per_thread(items_per_thread)
    threads_in_warp = _validate_logical_warp_threads(threads_in_warp)
    payload = (
        WarpMergeSortPayload.PAIRS
        if value_dtype is not None
        else WarpMergeSortPayload.KEYS
    )
    tile_policy = (
        WarpMergeSortTilePolicy.PARTIAL
        if valid_items is not None
        else WarpMergeSortTilePolicy.FULL
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
    if payload is WarpMergeSortPayload.PAIRS:
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
    if tile_policy is WarpMergeSortTilePolicy.PARTIAL:
        parameters.extend(
            (
                Value(INT32, name="valid_items"),
                Reference(_KEY_T, name="oob_default"),
            )
        )

    specialization = Algorithm(
        struct_name="WarpMergeSort",
        method_name="Sort",
        c_name="warp_merge_sort",
        includes=("cub/warp/warp_merge_sort.cuh",),
        template_parameters=(
            TemplateParameter("KeyT"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("VIRTUAL_WARP_THREADS"),
            TemplateParameter("ValueT"),
        ),
        parameters=(tuple(parameters),),
    ).specialize(
        {
            "KeyT": key_dtype,
            "ITEMS_PER_THREAD": items_per_thread,
            "VIRTUAL_WARP_THREADS": threads_in_warp,
            "ValueT": (
                value_dtype
                if payload is WarpMergeSortPayload.PAIRS
                else "::cub::NullType"
            ),
        },
        metadata={
            "scope": "warp",
            "primitive": "merge_sort",
            "payload": payload,
            "tile_policy": tile_policy,
            "operator": type(compare_operator).__qualname__,
        },
    )
    return WarpMergeSortSpec(
        specialization=specialization,
        payload=payload,
        tile_policy=tile_policy,
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        compare_operator=compare_operator,
    )
