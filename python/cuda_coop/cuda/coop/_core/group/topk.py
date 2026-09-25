# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral planning for block TopK keys and pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._bindings import ArgumentBinding, BindingKind
from .._symbols import semantic_token
from .._types import ArgumentKind, ParameterClassification, ParameterRole
from ..block.topk import make_block_topk_spec
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
class GroupTopKSemantics:
    key_dtype: Any
    items_per_thread: int
    selection: str
    k: ArgumentBinding
    value_dtype: Any = None
    valid_items: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)

    def __post_init__(self) -> None:
        if not isinstance(self.k, ArgumentBinding):
            raise TypeError("topk k must be an ArgumentBinding")
        if not isinstance(self.valid_items, ArgumentBinding):
            raise TypeError("topk valid_items must be an ArgumentBinding")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "topk",
            semantic_token(self.key_dtype),
            semantic_token(self.value_dtype),
            self.items_per_thread,
            self.selection,
            self.k.semantic_key,
            self.valid_items.semantic_key,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupTopKSemantics):
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
    result = [
        ParameterClassification("keys", ArgumentKind.RUNTIME, ParameterRole.INPUT)
    ]
    if operation.value_dtype is not None:
        result.append(
            ParameterClassification("values", ArgumentKind.RUNTIME, ParameterRole.INPUT)
        )
    for name, binding in (("k", operation.k), ("valid_items", operation.valid_items)):
        if binding.argument_kind is not None:
            result.append(
                ParameterClassification(
                    name,
                    binding.argument_kind,
                    ParameterRole.CONSTANT
                    if binding.argument_kind is ArgumentKind.STATIC
                    else ParameterRole.INPUT,
                )
            )
    return tuple(result)


def _plan_topk(call, resolved, launch, operation):
    spec = make_block_topk_spec(
        key_dtype=operation.key_dtype,
        value_dtype=operation.value_dtype,
        block_dim=launch.exact_block_dim,
        items_per_thread=operation.items_per_thread,
        selection=operation.selection,
        k=operation.k,
        num_valid=operation.valid_items,
    ).specialization
    results = [("keys", operation.key_dtype)]
    if operation.value_dtype is not None:
        results.append(("values", operation.value_dtype))
    result = ResultContract(
        tuple(
            LogicalResultContract(
                name=name,
                dtype=dtype,
                visibility=ResultVisibility.PER_MEMBER,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.ARRAY,
                items_per_member=operation.items_per_thread,
            )
            for name, dtype in results
        )
    )
    topology, participation, sync, storage = _contracts(
        resolved,
        launch,
        result=result,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=("k",)
        if operation.valid_items.kind is BindingKind.OMITTED
        else ("k", "valid_items"),
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
            header="cub/block/block_topk.cuh",
            cpp_class="cub::detail::block_topk",
            method=spec.method_name,
        ),
    )


_register_group_operation_family(
    GroupTopKSemantics,
    classifications=_classifications,
    planner=_plan_topk,
    group_kinds=frozenset({"block"}),
    unsupported_group_message="TopK supports only complete this_block() groups",
)
