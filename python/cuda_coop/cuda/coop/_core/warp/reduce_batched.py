# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bind CUB batched-reduction shapes for later backend materialization.

Each input slot identifies a separate reduction across lanes. The batch count
and warp width set the output-array extent; the layout chooses which lane owns
each aggregate. These descriptions do not compile code or allocate storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._algorithm import Algorithm
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


def _validate_logical_warp_threads(value: Any) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value not in {1, 2, 4, 8, 16, 32}
    ):
        raise ValueError(
            "threads_in_warp must be a power of two between 1 and 32"
        )
    return value


@dataclass(frozen=True, eq=False)
class WarpReduceBatchedSemantics:
    """Describe batches that each contain one input item from every lane.

    The dtype, batch count, operator, and layout form the semantic key, so
    calls with different reductions or result ownership get distinct
    identities. The enclosing group binds the warp width separately.

    Attributes
    ----------
    dtype : object
        Input and output element type accepted by the backend adapter. This
        record only checks that the type is present; it does not validate
        backend dtype support or introduce an accumulator type.
    batches : int
        Positive compile-time number of independent reductions, equal to the
        input-array extent in each lane. Boolean values are rejected.
    reduce_operator : CxxOperator or PythonOperator
        Binary operator descriptor. The operator must be associative and
        commutative and return the input dtype. These mathematical properties
        are the caller's responsibility. Stateful operators are unsupported.
    output_layout : {"striped", "blocked"}
        Assignment of batch aggregates to result slots, default ``"striped"``.
        This selects output ownership without changing the input-slot order.
    """

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
            raise ValueError(
                "reduce_batched output_layout must be striped or blocked"
            )

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
class WarpReduceBatchedSpecialization:
    """Pair a bound CUB call with the result shape needed by its caller.

    Attributes
    ----------
    specialization : Algorithm
        CUB method and bound template arguments, ready for backend
        materialization. Input and output are distinct array operands; the
        C++ method writes the output array instead of returning a value.
    call : WarpReduceBatchedSemantics
        Operation semantics before the group width was bound.
    threads_in_warp : int
        Participating lane count: 1, 2, 4, 8, 16, or 32.
    outputs_per_thread : int
        Output-array extent, ``ceil(call.batches / threads_in_warp)``. This is
        capacity; any slot without a corresponding batch is invalid.
    """

    specialization: Algorithm
    call: WarpReduceBatchedSemantics
    threads_in_warp: int
    outputs_per_thread: int


def make_warp_reduce_batched_specialization(
    *,
    dtype: Any,
    batches: int,
    threads_in_warp: int,
    reduce_operator: CxxOperator | PythonOperator,
    output_layout: str = "striped",
) -> WarpReduceBatchedSpecialization:
    """Select a CUB batched-reduction method and bind its array extents.

    This factory describes the call for a backend adapter. It does not compile
    the operator or wrapper, allocate result arrays, or allocate scratch. CUB
    synchronization is limited to the participating logical warp so other
    logical warps in the same physical warp can take a different branch.
    All lanes within the participating group must still enter the collective.

    Parameters
    ----------
    dtype : object
        Backend-neutral input and output element type. It must be present;
        the backend adapter is responsible for validating support.
    batches : int
        Positive compile-time input-array extent per lane. Each slot forms
        one independent reduction across the participating lanes.
    threads_in_warp : int
        Compile-time participating lane count: 1, 2, 4, 8, 16, or 32.
    reduce_operator : CxxOperator or PythonOperator
        Binary operator descriptor with the requirements documented by
        ``WarpReduceBatchedSemantics``. No stateful operator is accepted.
    output_layout : {"striped", "blocked"}, optional
        Default ``"striped"`` selects ``ReduceToStriped``; ``"blocked"``
        selects ``ReduceToBlocked``. Let ``W`` be the warp width and ``K``
        be ``ceil(batches / W)``. Slot ``i`` of lane ``r`` owns batch
        ``r + i * W`` for striped output or ``r * K + i`` for blocked output.

    Returns
    -------
    WarpReduceBatchedSpecialization
        Bound algorithm, operation semantics, warp width, and per-lane output
        capacity. Both arrays use ``dtype``. Output slots whose batch index
        is at least ``batches`` are invalid and must not be consumed.

    Raises
    ------
    ValueError
        The dtype is missing, the batch count is not a positive integer,
        the warp width is unsupported, or the output layout is unknown.
    TypeError
        The operator is not a C++ or stateless Python operator descriptor.
    """

    call = WarpReduceBatchedSemantics(
        dtype, batches, reduce_operator, output_layout
    )
    threads_in_warp = _validate_logical_warp_threads(threads_in_warp)
    outputs_per_thread = (batches + threads_in_warp - 1) // threads_in_warp
    specialization = Algorithm(
        struct_name="WarpReduceBatched",
        method_name=(
            "ReduceToStriped"
            if output_layout == "striped"
            else "ReduceToBlocked"
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
        template_arguments={
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
    return WarpReduceBatchedSpecialization(
        specialization, call, threads_in_warp, outputs_per_thread
    )


__all__ = [
    "WarpReduceBatchedSemantics",
    "WarpReduceBatchedSpecialization",
    "make_warp_reduce_batched_specialization",
]
