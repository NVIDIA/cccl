# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exchange semantics for block, physical-warp, and logical-warp groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._types import (
    ArgumentKind,
    ParameterClassification,
    ParameterRole,
)
from ..block.exchange import (
    BlockExchangeMode,
    BlockExchangeSemantics,
    BlockExchangeValueForm,
    make_block_exchange_spec,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ..warp.exchange import WarpExchangeMode, make_warp_exchange_spec
from ._contracts import _contracts, _unsupported, _unsupported_cub_warp_width
from ._dispatch import _register_group_operation_family
from ._model import (
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupPrimitiveCall,
    ImplementationProvenance,
    LogicalResultContract,
    ResultContract,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    UnsupportedReasonCode,
)

GroupExchangeMode = BlockExchangeMode
_BLOCK_WARP_STRIPED_MODES = frozenset(
    {
        BlockExchangeMode.WARP_STRIPED_TO_BLOCKED,
        BlockExchangeMode.BLOCKED_TO_WARP_STRIPED,
    }
)


@dataclass(frozen=True, eq=False)
class GroupExchangeSemantics:
    """Out-of-place Exchange operation selected after group resolution."""

    primitive: BlockExchangeSemantics

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockExchangeSemantics):
            raise TypeError("primitive must be BlockExchangeSemantics")
        if self.primitive.value_form is not BlockExchangeValueForm.OUT_OF_PLACE:
            raise ValueError("group exchange requires out-of-place value form")

    @property
    def dtype(self) -> Any:
        return self.primitive.dtype

    @property
    def mode(self) -> GroupExchangeMode:
        return GroupExchangeMode(self.primitive.mode.value)

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def result_visibility(self) -> ResultVisibility:
        return ResultVisibility.PER_MEMBER

    @property
    def returns_value(self) -> bool:
        return True

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupExchangeSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _call_classifications(
    operation: GroupExchangeSemantics,
) -> tuple[ParameterClassification, ...]:
    classifications = [
        ParameterClassification(
            "value",
            ArgumentKind.RUNTIME,
            ParameterRole.INPUT,
        )
    ]
    if operation.primitive.uses_ranks:
        classifications.append(
            ParameterClassification(
                "ranks",
                ArgumentKind.RUNTIME,
                ParameterRole.INPUT,
            )
        )
    if operation.primitive.uses_valid_flags:
        classifications.append(
            ParameterClassification(
                "valid_flags",
                ArgumentKind.RUNTIME,
                ParameterRole.INPUT,
            )
        )
    classifications.extend(
        (
            ParameterClassification(
                "mode",
                ArgumentKind.STATIC,
                ParameterRole.CONSTANT,
            ),
            ParameterClassification(
                "warp_time_slicing",
                ArgumentKind.STATIC,
                ParameterRole.CONSTANT,
            ),
        )
    )
    return tuple(classifications)


def _plan_exchange(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupExchangeSemantics,
) -> GroupLoweringPlan:
    assert launch.exact_block_dim is not None
    block_threads = launch.exact_block_threads
    assert block_threads is not None
    primitive = operation.primitive
    if resolved.kind == "block":
        if operation.mode in _BLOCK_WARP_STRIPED_MODES and block_threads % 32 != 0:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "CUB BlockExchange warp-striped modes require a block size "
                "that is a multiple of 32",
            )
        spec = make_block_exchange_spec(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=operation.items_per_thread,
            mode=operation.mode,
            value_form=BlockExchangeValueForm.OUT_OF_PLACE,
            warp_time_slicing=primitive.warp_time_slicing,
            rank_dtype=primitive.rank_dtype,
            valid_flag_dtype=primitive.valid_flag_dtype,
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        header = "cub/block/block_exchange.cuh"
    else:
        warp_width, width_error = _unsupported_cub_warp_width(call, resolved)
        if width_error is not None:
            return width_error
        assert warp_width is not None
        try:
            warp_mode = WarpExchangeMode(operation.mode.value)
        except ValueError:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                f"cub::WarpExchange does not support mode {operation.mode.value!r}",
            )
        if primitive.warp_time_slicing:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "warp_time_slicing applies only to BlockExchange",
            )
        spec = make_warp_exchange_spec(
            dtype=operation.dtype,
            items_per_thread=operation.items_per_thread,
            threads_in_warp=warp_width,
            mode=warp_mode,
            value_form=BlockExchangeValueForm.OUT_OF_PLACE.value,
            rank_dtype=primitive.rank_dtype,
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        header = "cub/warp/warp_exchange.cuh"

    result = ResultContract(
        (
            LogicalResultContract(
                name="value",
                dtype=operation.dtype,
                visibility=ResultVisibility.PER_MEMBER,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.ARRAY,
                items_per_member=operation.items_per_thread,
            ),
        )
    )
    contracts = _contracts(
        resolved,
        launch,
        result=result,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
    )
    return GroupLoweringPlan(
        target=target,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        topology=contracts[0],
        participation=contracts[1],
        result=result,
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=f"cub::{spec.struct_name}",
            method=spec.method_name,
        ),
    )


_register_group_operation_family(
    GroupExchangeSemantics,
    classifications=_call_classifications,
    planner=_plan_exchange,
    group_kinds=frozenset({"block", "warp", "threads_within_warp"}),
    unsupported_group_message=(
        "cuda.coop Exchange supports this_block(), complete physical "
        "this_warp(), and power-of-two logical-warp groups"
    ),
)


__all__ = ["GroupExchangeMode", "GroupExchangeSemantics"]
