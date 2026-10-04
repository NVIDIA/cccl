# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan fresh block histograms with an independent counter result shape.

The request records sample geometry and counting choices. Planning validates
those choices against the exact block dimensions, then describes the new
counter payload and the block's shared storage and synchronization. Frontends
use the result contract instead of assuming that output matches the input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._symbols import semantic_token
from .._types import INT32, ArgumentKind, ParameterClassification, ParameterRole
from ..block.histogram import make_block_histogram_specialization
from ._dispatch import _register_group_operation_family
from ._execution_requirements import _build_execution_requirements
from ._model import (
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    ImplementationProvenance,
    LogicalResultContract,
    ResultContract,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
)


@dataclass(frozen=True, eq=False)
class GroupHistogramSemantics:
    """Record histogram choices before exact launch dimensions are available.

    Sample dtype and items_per_thread describe the input. Counter dtype and
    bins_per_thread describe the separate result. bins is the number of real
    counters; extra output slots hold zero. The specialization builder
    validates these values during planning. The semantic key gives equal
    requests the same identity in every compiler.
    """

    sample_dtype: Any
    items_per_thread: int
    bins: int
    bins_per_thread: int = 1
    counter_dtype: Any = INT32
    algorithm: str = "atomic"

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "histogram",
            semantic_token(self.sample_dtype),
            self.items_per_thread,
            self.bins,
            self.bins_per_thread,
            semantic_token(self.counter_dtype),
            self.algorithm,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupHistogramSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def result_visibility(self) -> ResultVisibility:
        return ResultVisibility.PER_MEMBER

    @property
    def returns_value(self) -> bool:
        return True


def _classifications(operation):
    """Identify the samples as the group's only runtime input payload.

    Bin count, result extent, counter dtype and algorithm are specialization
    choices already stored in the operation record.
    """

    return (
        ParameterClassification(
            "samples", ArgumentKind.RUNTIME, ParameterRole.INPUT
        ),
    )


def _plan_histogram(call, resolved, launch, operation):
    """Specialize the block call and describe each member's returned counters.

    The builder checks block shape, supported dtypes, bin capacity and integer
    limits. The result contract uses counter dtype and bins_per_thread, not
    the input's dtype and extent. The shared contracts give the implementation
    ownership of scratch; a frontend replaces that for an explicit descriptor.
    Striped bin ownership is implemented by the C++ adapter.
    """

    specialization = make_block_histogram_specialization(
        sample_dtype=operation.sample_dtype,
        block_dim=launch.exact_block_dim,
        items_per_thread=operation.items_per_thread,
        bins=operation.bins,
        bins_per_thread=operation.bins_per_thread,
        counter_dtype=operation.counter_dtype,
        algorithm=operation.algorithm,
    ).specialization
    result = ResultContract(
        (
            LogicalResultContract(
                name="counts",
                dtype=operation.counter_dtype,
                visibility=ResultVisibility.PER_MEMBER,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.ARRAY,
                items_per_member=operation.bins_per_thread,
            ),
        )
    )
    requirements = _build_execution_requirements(
        resolved,
        launch,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=specialization,
        topology=requirements.topology,
        participation=requirements.participation,
        result=result,
        synchronization=requirements.synchronization,
        temp_storage=requirements.temp_storage,
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_histogram.cuh",
            cpp_class="cub::BlockHistogram",
            method="Histogram",
        ),
    )


_register_group_operation_family(
    GroupHistogramSemantics,
    classifications=_classifications,
    planner=_plan_histogram,
    group_kinds=frozenset({"block"}),
    unsupported_group_message=(
        "histogram supports only complete this_block() groups"
    ),
)
