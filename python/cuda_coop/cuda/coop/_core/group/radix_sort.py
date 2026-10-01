# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._types import INT32, ArgumentKind, ParameterClassification, ParameterRole
from ..block.radix_rank import (
    BlockRadixRankSemantics,
    make_block_radix_rank_specialization,
)
from ..block.radix_sort import (
    BlockRadixSortSemantics,
    make_block_radix_sort_specialization,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._dispatch import _register_group_operation_family
from ._execution_requirements import _build_execution_requirements
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
)


@dataclass(frozen=True)
class GroupRadixRankSemantics:
    """Stable digit ranks with an optional caller-provided prefix output."""

    primitive: BlockRadixRankSemantics
    operand_kind: GroupOperandKind = GroupOperandKind.ARRAY

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockRadixRankSemantics):
            raise TypeError("primitive must be BlockRadixRankSemantics")
        if not self.primitive.bit_range.is_static:
            raise ValueError(
                "group radix_rank_keys requires a static bit interval"
            )
        object.__setattr__(
            self, "operand_kind", GroupOperandKind(self.operand_kind)
        )
        if (
            self.operand_kind is GroupOperandKind.SCALAR
            and self.primitive.items_per_thread != 1
        ):
            raise ValueError(
                "scalar radix_rank_keys requires one item per thread"
            )

    @property
    def returns_value(self) -> bool:
        return True

    @property
    def result_visibility(self) -> ResultVisibility:
        return ResultVisibility.PER_MEMBER

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key, self.operand_kind.value


@dataclass(frozen=True)
class GroupRadixSortSemantics:
    """Nonmutating sort results backed by an in-place CUB specialization."""

    primitive: BlockRadixSortSemantics
    operand_kind: GroupOperandKind = GroupOperandKind.ARRAY

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockRadixSortSemantics):
            raise TypeError("primitive must be BlockRadixSortSemantics")
        object.__setattr__(
            self, "operand_kind", GroupOperandKind(self.operand_kind)
        )
        if (
            self.operand_kind is GroupOperandKind.SCALAR
            and self.primitive.items_per_thread != 1
        ):
            raise ValueError("scalar radix sort requires one item per thread")

    @property
    def returns_value(self) -> bool:
        return True

    @property
    def result_visibility(self) -> ResultVisibility:
        return ResultVisibility.PER_MEMBER

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key, self.operand_kind.value


def _classifications(
    operation: GroupRadixRankSemantics | GroupRadixSortSemantics,
) -> tuple[ParameterClassification, ...]:
    p = operation.primitive
    fields = [
        ParameterClassification(
            "keys", ArgumentKind.RUNTIME, ParameterRole.INPUT
        )
    ]
    if isinstance(operation, GroupRadixSortSemantics) and p.has_values:
        fields.append(
            ParameterClassification(
                "values", ArgumentKind.RUNTIME, ParameterRole.INPUT
            )
        )
    if (
        isinstance(operation, GroupRadixRankSemantics)
        and p.has_exclusive_digit_prefix
    ):
        fields.append(
            ParameterClassification(
                "exclusive_digit_prefix",
                ArgumentKind.RUNTIME,
                ParameterRole.OUTPUT,
            )
        )
    fields.append(
        ParameterClassification(
            "descending", ArgumentKind.STATIC, ParameterRole.CONSTANT
        )
    )
    if p.bit_range is not None:
        for name in ("begin_bit", "end_bit"):
            binding = getattr(p.bit_range, name)
            kind = binding.argument_kind
            fields.append(
                ParameterClassification(
                    name,
                    kind,
                    ParameterRole.CONSTANT
                    if kind is ArgumentKind.STATIC
                    else ParameterRole.INPUT,
                )
            )
    return tuple(fields)


def _plan(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupRadixRankSemantics | GroupRadixSortSemantics,
) -> GroupLoweringPlan:
    p = operation.primitive
    kwargs = {
        "key_dtype": p.key_dtype,
        "block_dim": launch.exact_block_dim,
        "items_per_thread": p.items_per_thread,
        "descending": p.descending,
    }
    if isinstance(operation, GroupRadixRankSemantics):
        if (
            p.block_threads is not None
            and p.block_threads != launch.exact_block_threads
        ):
            raise ValueError(
                "radix_rank_keys block_threads disagrees with the launch"
            )
        specialization = make_block_radix_rank_specialization(
            **kwargs,
            begin_bit=p.bit_range.static_begin_bit,
            end_bit=p.bit_range.static_end_bit,
            key_bit_width=p.bit_range.bit_width,
            with_exclusive_digit_prefix=p.has_exclusive_digit_prefix,
        ).specialization
        results = (("ranks", INT32),)
        header = "cub/block/block_radix_rank.cuh"
    else:
        kwargs.update(
            value_dtype=p.value_dtype,
            blocked_to_striped=p.blocked_to_striped,
            bit_policy=p.bit_policy,
        )
        if p.bit_range is not None:
            kwargs.update(
                begin_bit=p.bit_range.begin_bit,
                end_bit=p.bit_range.end_bit,
                key_bit_width=p.bit_range.bit_width,
            )
        specialization = make_block_radix_sort_specialization(
            **kwargs
        ).specialization
        results = (("keys", p.key_dtype),) + (
            (("values", p.value_dtype),) if p.has_values else ()
        )
        header = "cub/block/block_radix_sort.cuh"
    result = ResultContract(
        tuple(
            LogicalResultContract(
                name=name,
                dtype=dtype,
                visibility=ResultVisibility.PER_MEMBER,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=operation.operand_kind,
                items_per_member=p.items_per_thread,
            )
            for name, dtype in results
        )
    )
    requirements = _build_execution_requirements(
        resolved,
        launch,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=("begin_bit", "end_bit"),
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
            header=header,
            cpp_class="cub::BlockRadixRank"
            if isinstance(operation, GroupRadixRankSemantics)
            else "cub::BlockRadixSort",
            method=specialization.method_name,
        ),
    )


for _semantics in (GroupRadixRankSemantics, GroupRadixSortSemantics):
    _register_group_operation_family(
        _semantics,
        classifications=_classifications,
        planner=_plan,
        group_kinds=frozenset({"block"}),
        unsupported_group_message=(
            "cuda.coop radix operations require a complete physical block"
        ),
    )
del _semantics

__all__ = [
    "GroupRadixRankSemantics",
    "GroupRadixSortSemantics",
]
