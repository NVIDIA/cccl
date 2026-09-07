# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exchange semantics and lowering for explicit thread groups.

This module maps the portable exchange modes to CUB block or warp
specializations after group resolution. It deliberately does not own public
argument capture or backend compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..block.exchange import (
    BlockExchangeMode,
    BlockExchangeSemantics,
    BlockExchangeValueForm,
    make_block_exchange_spec,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ..warp.exchange import (
    WarpExchangeMode,
    WarpExchangeValueForm,
    make_warp_exchange_spec,
)
from ._contracts import _contracts, _unsupported, _unsupported_cub_warp_width
from ._model import (
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupPrimitiveCall,
    ImplementationProvenance,
    ResultVisibility,
    StorageOwnership,
    UnsupportedReasonCode,
)

GroupExchangeMode = BlockExchangeMode


@dataclass(frozen=True, eq=False)
class GroupExchangeSemantics:
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
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupExchangeSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _plan_exchange(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupExchangeSemantics,
) -> GroupLoweringPlan:
    if resolved.kind not in {"block", "warp", "threads_within_warp"}:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group exchange supports physical block, physical-warp, and "
            "logical-warp groups",
        )
    assert launch.exact_block_dim is not None
    if resolved.kind == "block":
        spec = make_block_exchange_spec(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=operation.items_per_thread,
            mode=BlockExchangeMode(operation.mode.value),
            value_form=BlockExchangeValueForm.OUT_OF_PLACE,
            warp_time_slicing=operation.primitive.warp_time_slicing,
            rank_dtype=operation.primitive.rank_dtype,
            valid_flag_dtype=operation.primitive.valid_flag_dtype,
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        cpp_class = "cub::BlockExchange"
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
        if operation.primitive.warp_time_slicing:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "warp_time_slicing applies only to BlockExchange",
            )
        if operation.primitive.valid_flag_dtype is not None:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "cub::WarpExchange does not accept valid_flags",
            )
        spec = make_warp_exchange_spec(
            dtype=operation.dtype,
            items_per_thread=operation.items_per_thread,
            threads_in_warp=warp_width,
            mode=warp_mode,
            value_form=WarpExchangeValueForm.OUT_OF_PLACE,
            rank_dtype=operation.primitive.rank_dtype,
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        cpp_class = "cub::WarpExchange"
        header = "cub/warp/warp_exchange.cuh"
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
    )
    return GroupLoweringPlan(
        target=target,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=cpp_class,
            method=spec.method_name,
        ),
    )


__all__ = ["GroupExchangeMode", "GroupExchangeSemantics"]
