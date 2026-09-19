# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral independent-batch reductions across a warp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._symbols import semantic_token
from .._types import (
    Array,
    CxxOperator,
    Dependency,
    PythonOperator,
    TemplateParameter,
    TempStorageParameter,
)
from ..block._common import normalize_positive_int
from .reduce import _validate_logical_warp_threads


@dataclass(frozen=True, eq=False)
class WarpReduceBatchedSemantics:
    """Each payload slot is a separate batch containing one item per lane."""

    dtype: Any
    batches: int
    reduce_operator: CxxOperator | PythonOperator
    output_layout: str = "striped"

    def __post_init__(self) -> None:
        if self.dtype is None:
            raise ValueError("reduce_batched dtype must be provided")
        normalize_positive_int("batches", self.batches)
        if not isinstance(self.reduce_operator, (CxxOperator, PythonOperator)):
            raise TypeError("reduce_batched requires a reduction operator")
        if self.output_layout not in {"striped", "blocked"}:
            raise ValueError("reduce_batched output_layout must be striped or blocked")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "reduce_batched",
            semantic_token(self.dtype),
            self.batches,
            semantic_token(self.reduce_operator),
            self.output_layout,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WarpReduceBatchedSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True)
class WarpReduceBatchedSpec:
    specialization: AlgorithmSpec
    call: WarpReduceBatchedSemantics
    threads_in_warp: int
    outputs_per_thread: int


def make_warp_reduce_batched_spec(
    *,
    dtype: Any,
    batches: int,
    threads_in_warp: int,
    reduce_operator: CxxOperator | PythonOperator,
    output_layout: str = "striped",
) -> WarpReduceBatchedSpec:
    """Use CUB's native batched collective and its distributed output layout."""

    call = WarpReduceBatchedSemantics(dtype, batches, reduce_operator, output_layout)
    threads_in_warp = _validate_logical_warp_threads(threads_in_warp)
    outputs_per_thread = (batches + threads_in_warp - 1) // threads_in_warp
    specialization = Algorithm(
        struct_name="WarpReduceBatched",
        method_name=(
            "ReduceToStriped" if output_layout == "striped" else "ReduceToBlocked"
        ),
        c_name="warp_reduce_batched",
        includes=(
            "cub/warp/warp_reduce_batched.cuh",
            "cuda/std/functional",
            "cuda/functional",
        ),
        template_parameters=(
            TemplateParameter("T"),
            TemplateParameter("BATCHES"),
            TemplateParameter("LOGICAL_WARP_THREADS"),
            TemplateParameter("SYNC_PHYSICAL_WARP"),
        ),
        parameters=(
            (
                TempStorageParameter(),
                Array(Dependency("T"), Dependency("BATCHES"), name="input"),
                Array(
                    Dependency("T"),
                    Dependency("OUTPUTS_PER_THREAD"),
                    name="output",
                    is_output=True,
                    is_return=False,
                ),
                reduce_operator,
            ),
        ),
    ).specialize(
        {
            "T": dtype,
            "BATCHES": batches,
            "LOGICAL_WARP_THREADS": threads_in_warp,
            # Synchronize only the participating logical warp. Requiring other
            # logical warps to enter a divergent branch would risk deadlock.
            "SYNC_PHYSICAL_WARP": "false",
            "OUTPUTS_PER_THREAD": outputs_per_thread,
        },
        metadata={
            "scope": "warp",
            "primitive": "reduce_batched",
            "output_layout": output_layout,
            "operator": semantic_token(reduce_operator),
        },
    )
    return WarpReduceBatchedSpec(
        specialization, call, threads_in_warp, outputs_per_thread
    )


__all__ = [
    "WarpReduceBatchedSemantics",
    "WarpReduceBatchedSpec",
    "make_warp_reduce_batched_spec",
]
