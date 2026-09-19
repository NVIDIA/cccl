# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral complete-operation Run Length Decode planning."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from .._bindings import ArgumentBinding
from .._symbols import semantic_token
from .._types import (
    UINT32,
    UINT64,
    ArgumentKind,
    ParameterClassification,
    ParameterRole,
)
from ..block.run_length import make_block_run_length_decode_spec
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
class GroupRunLengthDecodeSemantics:
    item_dtype: Any
    run_length_dtype: Any
    runs_per_thread: int
    decoded_items_per_thread: int
    offset: ArgumentBinding = ArgumentBinding.static(0)
    decoded_offset_dtype: Any = UINT32
    control_dtype: Any = UINT64
    bulk: bool = False
    relative_offsets: bool = False

    @property
    def semantic_key(self):
        return (
            "run_length_decode",
            *(semantic_token(getattr(self, field.name)) for field in fields(self)),
        )

    def __eq__(self, other):
        if not isinstance(other, GroupRunLengthDecodeSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self):
        return hash(self.semantic_key)

    @property
    def result_visibility(self):
        return (
            ResultVisibility.ALL_MEMBERS if self.bulk else ResultVisibility.PER_MEMBER
        )

    @property
    def returns_value(self):
        return True


def _classifications(operation):
    names = ["run_values", "run_lengths"]
    if operation.bulk:
        names.append("destination")
        if operation.relative_offsets:
            names.append("relative_offsets")
    result = [
        ParameterClassification(
            name,
            ArgumentKind.RUNTIME,
            ParameterRole.INPUT if name.startswith("run_") else ParameterRole.OUTPUT,
        )
        for name in names
    ]
    result.append(
        ParameterClassification(
            "destination_offset" if operation.bulk else "decoded_window_offset",
            operation.offset.argument_kind,
            ParameterRole.CONSTANT
            if operation.offset.argument_kind is ArgumentKind.STATIC
            else ParameterRole.INPUT,
        )
    )
    return tuple(result)


def _plan(call, resolved, launch, operation):
    spec = make_block_run_length_decode_spec(
        **{field.name: getattr(operation, field.name) for field in fields(operation)},
        block_dim=launch.exact_block_dim,
    ).specialization
    result = ResultContract(
        (
            LogicalResultContract(
                name="total_decoded_size" if operation.bulk else "decoded",
                dtype=operation.decoded_offset_dtype
                if operation.bulk
                else operation.item_dtype,
                visibility=operation.result_visibility,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.SCALAR
                if operation.bulk
                else GroupOperandKind.ARRAY,
                items_per_member=1
                if operation.bulk
                else operation.decoded_items_per_thread,
            ),
        )
    )
    topology, participation, sync, storage = _contracts(
        resolved,
        launch,
        result=result,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=(
            ("destination", "destination_offset", "relative_offsets")
            if operation.relative_offsets
            else ("destination", "destination_offset")
        )
        if operation.bulk
        else ("decoded_window_offset",),
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
            header="cub/block/block_run_length_decode.cuh",
            cpp_class="cub::BlockRunLengthDecode",
            method="RunLengthDecode",
        ),
    )


_register_group_operation_family(
    GroupRunLengthDecodeSemantics,
    classifications=_classifications,
    planner=_plan,
    group_kinds=frozenset({"block"}),
    unsupported_group_message="run_length_decode requires a complete this_block() group",
)
