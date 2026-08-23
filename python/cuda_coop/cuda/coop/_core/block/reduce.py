# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockReduce semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._bindings import ArgumentBinding, BindingKind
from .._types import (
    INT32,
    Array,
    CxxFunction,
    CxxOperator,
    Dependency,
    PythonOperator,
    Reference,
    StatefulOperator,
    TemplateParameter,
    TempStorageParameter,
    Value,
)
from ..reduce import (
    ReduceOperation,
    ReduceSemantics,
    ReduceValueKind,
    make_reduce_semantics,
)
from ._common import normalize_block_dim

BlockReduceOperation = ReduceOperation
BlockReduceSemantics = ReduceSemantics
BlockReduceValueKind = ReduceValueKind
make_block_reduce_semantics = make_reduce_semantics


class BlockReduceAlgorithm(str, Enum):
    """Public CUB ``BlockReduceAlgorithm`` enumerators."""

    RAKING_COMMUTATIVE_ONLY = "::cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY"
    RAKING = "::cub::BLOCK_REDUCE_RAKING"
    WARP_REDUCTIONS = "::cub::BLOCK_REDUCE_WARP_REDUCTIONS"
    WARP_REDUCTIONS_NONDETERMINISTIC = (
        "::cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC"
    )


def normalize_block_reduce_algorithm(
    algorithm: str | BlockReduceAlgorithm,
) -> BlockReduceAlgorithm:
    if isinstance(algorithm, BlockReduceAlgorithm):
        return algorithm
    for candidate in BlockReduceAlgorithm:
        scoped = candidate.value.replace(
            "::cub::",
            "::cub::BlockReduceAlgorithm::",
            1,
        )
        if algorithm in {candidate.value, scoped}:
            return candidate
    raise ValueError(f"unsupported CUB BlockReduce algorithm {algorithm!r}")


@dataclass(frozen=True)
class BlockReduceSpec:
    """Fully specialized CUB BlockReduce semantics."""

    specialization: AlgorithmSpec
    call: BlockReduceSemantics
    block_dim: tuple[int, int, int]
    algorithm: BlockReduceAlgorithm

    @property
    def operation(self) -> BlockReduceOperation:
        return self.call.operation

    @property
    def value_kind(self) -> BlockReduceValueKind:
        return self.call.value_kind

    @property
    def items_per_thread(self) -> int:
        return self.call.items_per_thread

    @property
    def has_valid_items(self) -> bool:
        return self.call.has_valid_items

    @property
    def valid_items(self) -> ArgumentBinding:
        return self.call.valid_items

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def _block_reduce_parameters(call: ReduceSemantics) -> tuple[Any, ...]:
    parameters: list[Any] = [TempStorageParameter()]
    if call.value_kind is BlockReduceValueKind.ARRAY:
        parameters.append(
            Array(
                Dependency("T"),
                Dependency("ITEMS_PER_THREAD"),
                name="src",
            )
        )
    else:
        parameters.append(Reference(Dependency("T"), name="src"))
    if call.reduce_operator is not None:
        parameters.append(call.reduce_operator)
    if call.valid_items.kind is BindingKind.RUNTIME:
        parameters.append(Value(INT32, name="num_valid"))
    elif call.valid_items.kind is BindingKind.STATIC:
        parameters.append(
            CxxFunction(str(call.valid_items.value), INT32, name="num_valid")
        )
    parameters.append(
        Reference(
            Dependency("T"),
            name="output",
            is_output=True,
            is_return=True,
        )
    )

    return tuple(parameters)


def make_block_reduce_spec(
    *,
    dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    operation: str | BlockReduceOperation,
    algorithm: str | BlockReduceAlgorithm,
    value_kind: str | BlockReduceValueKind,
    reduce_operator: CxxOperator | PythonOperator | StatefulOperator | None = None,
    valid_items: bool | ArgumentBinding = False,
) -> BlockReduceSpec:
    """Build a fully specialized CUB BlockReduce description."""

    block_dim = normalize_block_dim(block_dim)
    algorithm = normalize_block_reduce_algorithm(algorithm)
    call = make_block_reduce_semantics(
        dtype=dtype,
        items_per_thread=items_per_thread,
        operation=operation,
        value_kind=value_kind,
        reduce_operator=reduce_operator,
        valid_items=valid_items,
    )
    if call.valid_items.kind is BindingKind.STATIC:
        static_valid_items = call.valid_items.value
        assert isinstance(static_valid_items, int)
        block_threads = block_dim[0] * block_dim[1] * block_dim[2]
        if static_valid_items > block_threads:
            raise ValueError(
                f"static valid_items {static_valid_items} exceeds block size "
                f"{block_threads}"
            )

    template_arguments = {
        "T": dtype,
        "BLOCK_DIM_X": block_dim[0],
        "ALGORITHM": algorithm.value,
        "BLOCK_DIM_Y": block_dim[1],
        "BLOCK_DIM_Z": block_dim[2],
    }
    if call.value_kind is BlockReduceValueKind.ARRAY:
        template_arguments["ITEMS_PER_THREAD"] = items_per_thread

    specialization = Algorithm(
        struct_name="BlockReduce",
        method_name=call.method_name,
        c_name="block_reduce",
        includes=("cub/block/block_reduce.cuh",),
        template_parameters=(
            TemplateParameter("T"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("ALGORITHM"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ),
        parameters=(_block_reduce_parameters(call),),
    ).specialize(
        template_arguments,
        metadata={
            "scope": "block",
            "primitive": "reduce",
            "operation": call.operation,
            "value_kind": call.value_kind,
            "valid_items": call.has_valid_items,
            "operator": (
                None
                if call.reduce_operator is None
                else type(call.reduce_operator).__qualname__
            ),
        },
    )
    return BlockReduceSpec(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
        algorithm=algorithm,
    )
