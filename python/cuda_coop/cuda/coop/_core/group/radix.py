# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Radix-sort and radix-rank semantics with complete-block lowering.

The shared radix family keeps bit-range and output contracts together because
both planners enforce the same CUB tile constraints. Frontend validation,
compiler activation, and generated-provider lifecycle remain backend-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._symbols import semantic_token
from .._types import INT32, RuntimeValue
from ..block.radix_rank import BlockRadixRankSemantics, make_block_radix_rank_spec
from ..block.radix_sort import (
    BlockRadixSortBitPolicy,
    BlockRadixSortOutput,
    BlockRadixSortSemantics,
    make_block_radix_sort_spec,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._contracts import _contracts, _unsupported
from ._model import (
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupPrimitiveCall,
    ImplementationProvenance,
    LogicalResultContract,
    ParticipationContract,
    ResultContract,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    SynchronizationContract,
    TempStorageContract,
    UnsupportedReasonCode,
)


@dataclass(frozen=True, eq=False)
class GroupRadixSortSemantics:
    """Block-radix-sort semantics attached to an explicit group."""

    primitive: BlockRadixSortSemantics
    operand_kind: GroupOperandKind = GroupOperandKind.ARRAY

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockRadixSortSemantics):
            raise TypeError("primitive must be BlockRadixSortSemantics")
        object.__setattr__(self, "operand_kind", GroupOperandKind(self.operand_kind))
        if self.primitive.bit_policy is not BlockRadixSortBitPolicy.EXPLICIT:
            raise ValueError("group radix sort requires explicit runtime bit bounds")
        if self.primitive.output is not BlockRadixSortOutput.BLOCKED:
            raise ValueError("group radix sort requires blocked output")
        if (
            self.operand_kind is GroupOperandKind.SCALAR
            and self.primitive.items_per_thread != 1
        ):
            raise ValueError("scalar group radix sort requires one item per thread")

    @property
    def dtype(self) -> Any:
        return self.primitive.key_dtype

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key, self.operand_kind.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupRadixSortSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupRadixRankSemantics:
    """Block-radix-rank semantics attached to an explicit group.

    ``primitive.key_dtype`` is the unsigned bit-ordered type consumed by CUB.
    ``input_dtype`` records the public key type before signed keys are adapted
    to that representation.
    """

    primitive: BlockRadixRankSemantics
    input_dtype: Any
    operand_kind: GroupOperandKind = GroupOperandKind.ARRAY

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockRadixRankSemantics):
            raise TypeError("primitive must be BlockRadixRankSemantics")
        if self.input_dtype is None:
            raise ValueError("input_dtype must be provided")
        object.__setattr__(self, "operand_kind", GroupOperandKind(self.operand_kind))
        if not self.primitive.bit_range.is_static:
            raise ValueError("group radix rank requires a static radix bit range")
        if (
            self.operand_kind is GroupOperandKind.SCALAR
            and self.primitive.items_per_thread != 1
        ):
            raise ValueError("scalar group radix rank requires one item per thread")

    @property
    def dtype(self) -> Any:
        return self.input_dtype

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.primitive.semantic_key,
            semantic_token(self.input_dtype),
            self.operand_kind.value,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupRadixRankSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


_CUB_RADIX_MAX_TILE_ITEMS = (1 << 16) - 1


def _radix_result(
    *,
    name: str,
    dtype: Any,
    operand_kind: GroupOperandKind,
    items_per_member: int,
) -> LogicalResultContract:
    return LogicalResultContract(
        name=name,
        dtype=dtype,
        visibility=ResultVisibility.PER_MEMBER,
        ownership=ResultOwnership.EACH_MEMBER,
        operand_kind=operand_kind,
        items_per_member=items_per_member,
    )


def _radix_contracts(
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupRadixSortSemantics | GroupRadixRankSemantics,
    *,
    uniform_arguments: tuple[str, ...] = (),
) -> tuple[
    ParticipationContract,
    SynchronizationContract,
    TempStorageContract,
]:
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=uniform_arguments,
        returns_value=False,
    )
    return contracts[0], contracts[2], contracts[3]


def _radix_tile_failure(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    *,
    block_threads: int,
    items_per_thread: int,
) -> GroupLoweringPlan | None:
    tile_items = block_threads * items_per_thread
    if tile_items <= _CUB_RADIX_MAX_TILE_ITEMS:
        return None
    return _unsupported(
        call,
        resolved,
        UnsupportedReasonCode.OPERATION_VARIANT,
        "CUB block radix collectives require block_threads * items_per_thread "
        f"<= {_CUB_RADIX_MAX_TILE_ITEMS}; received {tile_items}",
    )


def _plan_radix_sort(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupRadixSortSemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group radix sort supports complete physical block groups",
        )
    assert launch.exact_block_dim is not None
    assert launch.exact_block_threads is not None
    failure = _radix_tile_failure(
        call,
        resolved,
        block_threads=launch.exact_block_threads,
        items_per_thread=operation.items_per_thread,
    )
    if failure is not None:
        return failure
    primitive = operation.primitive
    bit_width = None if primitive.bit_range is None else primitive.bit_range.bit_width
    spec = make_block_radix_sort_spec(
        key_dtype=primitive.key_dtype,
        value_dtype=primitive.value_dtype,
        block_dim=launch.exact_block_dim,
        items_per_thread=primitive.items_per_thread,
        descending=primitive.order,
        blocked_to_striped=False,
        begin_bit=RuntimeValue("begin_bit"),
        end_bit=RuntimeValue("end_bit"),
        key_bit_width=bit_width,
        bit_policy=BlockRadixSortBitPolicy.EXPLICIT,
    ).specialization
    participation, synchronization, temp_storage = _radix_contracts(
        resolved,
        launch,
        operation,
        uniform_arguments=("begin_bit", "end_bit"),
    )
    result_values = [
        _radix_result(
            name="keys",
            dtype=primitive.key_dtype,
            operand_kind=operation.operand_kind,
            items_per_member=primitive.items_per_thread,
        )
    ]
    if primitive.has_values:
        result_values.append(
            _radix_result(
                name="values",
                dtype=primitive.value_dtype,
                operand_kind=operation.operand_kind,
                items_per_member=primitive.items_per_thread,
            )
        )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=participation,
        result=ResultContract(tuple(result_values)),
        synchronization=synchronization,
        temp_storage=temp_storage,
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_radix_sort.cuh",
            cpp_class="cub::BlockRadixSort",
            method=spec.method_name,
        ),
    )


def _plan_radix_rank(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupRadixRankSemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group radix rank supports complete physical block groups",
        )
    assert launch.exact_block_dim is not None
    assert launch.exact_block_threads is not None
    failure = _radix_tile_failure(
        call,
        resolved,
        block_threads=launch.exact_block_threads,
        items_per_thread=operation.items_per_thread,
    )
    if failure is not None:
        return failure
    primitive = operation.primitive
    begin_bit = primitive.bit_range.static_begin_bit
    end_bit = primitive.bit_range.static_end_bit
    assert begin_bit is not None and end_bit is not None
    spec = make_block_radix_rank_spec(
        key_dtype=primitive.key_dtype,
        block_dim=launch.exact_block_dim,
        items_per_thread=primitive.items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_bit_width=primitive.bit_range.bit_width,
        descending=primitive.order,
        with_exclusive_digit_prefix=primitive.has_exclusive_digit_prefix,
    ).specialization
    participation, synchronization, temp_storage = _radix_contracts(
        resolved,
        launch,
        operation,
    )
    result_values = [
        _radix_result(
            name="ranks",
            dtype=INT32,
            operand_kind=operation.operand_kind,
            items_per_member=primitive.items_per_thread,
        )
    ]
    if primitive.has_exclusive_digit_prefix:
        prefix_items = primitive.exclusive_digit_prefix_items_per_thread
        assert prefix_items is not None
        result_values.append(
            _radix_result(
                name="exclusive_digit_prefix",
                dtype=INT32,
                operand_kind=GroupOperandKind.ARRAY,
                items_per_member=prefix_items,
            )
        )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=participation,
        result=ResultContract(tuple(result_values)),
        synchronization=synchronization,
        temp_storage=temp_storage,
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_radix_rank.cuh",
            cpp_class="cub::BlockRadixRank",
            method=spec.method_name,
        ),
    )


__all__ = ["GroupRadixRankSemantics", "GroupRadixSortSemantics"]
