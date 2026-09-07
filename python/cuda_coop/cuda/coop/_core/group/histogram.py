# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Histogram semantics and complete-block lowering.

The family validates the static output layout and lowers a resolved block to
the public CUB BlockHistogram primitive. It does not own frontend payload
inference or backend provider generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..block.histogram import (
    BlockHistogramOperation,
    BlockHistogramSemantics,
    make_block_histogram_spec,
    validate_block_histogram_output_capacity,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._contracts import _contracts, _unsupported
from ._model import (
    ArgumentPrecondition,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupPrimitiveCall,
    ImplementationProvenance,
    PreconditionEnforcement,
    ResultVisibility,
    StorageOwnership,
    UnsupportedReasonCode,
)


@dataclass(frozen=True, eq=False)
class GroupHistogramSemantics:
    """Static-width BlockHistogram semantics attached to an explicit group."""

    primitive: BlockHistogramSemantics
    bins_per_thread: int

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockHistogramSemantics):
            raise TypeError("primitive must be BlockHistogramSemantics")
        if not self.primitive.has_static_bins:
            raise ValueError("group histogram requires a static bin count")
        if self.primitive.operation is not BlockHistogramOperation.HISTOGRAM:
            raise ValueError("group histogram requires the public Histogram operation")
        if (
            not isinstance(self.bins_per_thread, int)
            or isinstance(self.bins_per_thread, bool)
            or self.bins_per_thread < 1
        ):
            raise ValueError("bins_per_thread must be a positive integer")

    @property
    def dtype(self) -> Any:
        """Result counter dtype consumed by the common result contract."""

        return self.primitive.counter_dtype

    @property
    def items_per_thread(self) -> int:
        """Number of striped histogram counters returned to each member."""

        return self.bins_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key, self.bins_per_thread

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupHistogramSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _plan_histogram(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupHistogramSemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group histogram supports complete physical block groups",
        )
    assert launch.exact_block_dim is not None
    primitive = operation.primitive
    assert primitive.bins is not None
    assert resolved.static_size is not None
    validate_block_histogram_output_capacity(
        bins=primitive.bins,
        bins_per_thread=operation.bins_per_thread,
        block_threads=resolved.static_size,
    )
    spec = make_block_histogram_spec(
        item_dtype=primitive.item_dtype,
        counter_dtype=primitive.counter_dtype,
        block_dim=launch.exact_block_dim,
        items_per_thread=primitive.items_per_thread,
        bins=primitive.bins,
        algorithm=primitive.algorithm,
        operation=BlockHistogramOperation.HISTOGRAM,
    ).specialization
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        argument_preconditions=(
            ArgumentPrecondition(
                name="samples",
                minimum=0,
                maximum=primitive.bins - 1,
                enforcement=PreconditionEnforcement.CALLER,
            ),
        ),
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_histogram.cuh",
            cpp_class="cub::BlockHistogram",
            method=spec.method_name,
        ),
    )


__all__ = ["GroupHistogramSemantics"]
