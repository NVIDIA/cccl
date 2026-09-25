# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral planning for fresh block histograms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._symbols import semantic_token
from .._types import INT32, ArgumentKind, ParameterClassification, ParameterRole
from ..block.histogram import make_block_histogram_spec
from ._contracts import _contracts
from ._dispatch import _register_group_operation_family
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
    return (
        ParameterClassification("samples", ArgumentKind.RUNTIME, ParameterRole.INPUT),
    )


def _plan_histogram(call, resolved, launch, operation):
    spec = make_block_histogram_spec(
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
    topology, participation, sync, storage = _contracts(
        resolved,
        launch,
        result=result,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        topology=topology,
        participation=participation,
        result=result,
        synchronization=sync,
        temp_storage=storage,
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
    unsupported_group_message="histogram supports only complete this_block() groups",
)
