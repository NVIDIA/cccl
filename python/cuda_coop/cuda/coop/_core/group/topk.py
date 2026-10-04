# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan block TopK results, participation, and scratch ownership.

Count bindings distinguish compile-time values from runtime inputs. The plan
retains each payload's full extent while the public TopK contract limits reads
to the selected prefix. Compiler adapters preserve inputs with working
copies. By default, lowering arranges scratch; an explicit TempStorage makes
it caller-owned. Importing this module registers TopK for block groups.
Planning does not allocate memory or run the call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._bindings import ArgumentBinding, BindingKind
from .._symbols import semantic_token
from .._types import ArgumentKind, ParameterClassification, ParameterRole
from ..block.topk import make_block_topk_specialization
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
class GroupTopKSemantics:
    """Describe a TopK request independently of its block launch shape.

    ``key_dtype`` and optional ``value_dtype`` identify keys-only or paired
    selection. ``items_per_thread`` fixes the payload extent; ``selection``
    chooses minimum or maximum keys. ``k`` is a count binding, and omission of
    ``valid_items`` means the full tile. Count bindings are part of semantic
    identity so different static counts do not share the same request key.

    Construction checks the binding objects. The block specialization later
    validates dimensions, extent, direction, and known count values.
    """

    key_dtype: Any
    items_per_thread: int
    selection: str
    k: ArgumentBinding
    value_dtype: Any = None
    valid_items: ArgumentBinding = field(
        default_factory=ArgumentBinding.omitted
    )

    def __post_init__(self) -> None:
        """Keep static and runtime counts in explicit binding objects."""

        if not isinstance(self.k, ArgumentBinding):
            raise TypeError("topk k must be an ArgumentBinding")
        if not isinstance(self.valid_items, ArgumentBinding):
            raise TypeError("topk valid_items must be an ArgumentBinding")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """Key the request by dtypes, shape, direction, and count bindings."""

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
        """Report per-thread results; only the selected prefix is defined."""

        return ResultVisibility.PER_MEMBER

    @property
    def returns_value(self) -> bool:
        """Report that every TopK request produces result payloads."""

        return True


def _classifications(operation):
    """Classify payloads as inputs and counts by their binding policy.

    Omitted counts need no call operand. Static counts are constants; runtime
    counts remain inputs that the backend passes to the specialized operation.
    """

    result = [
        ParameterClassification(
            "keys", ArgumentKind.RUNTIME, ParameterRole.INPUT
        )
    ]
    if operation.value_dtype is not None:
        result.append(
            ParameterClassification(
                "values", ArgumentKind.RUNTIME, ParameterRole.INPUT
            )
        )
    for name, binding in (
        ("k", operation.k),
        ("valid_items", operation.valid_items),
    ):
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
    """Select CUB TopK and describe each thread's result and shared scratch.

    The dispatcher admits only complete block groups. Specialization uses the
    exact launch dimensions and rejects unsupported shapes or invalid static
    counts. Record one key result and an optional value result, each
    retaining its input dtype and per-thread extent. These extents do not
    make the tail outside ``min(k, valid_items)`` readable.

    Require every member to use the same counts. Omitted ``valid_items`` is
    already fixed by the tile shape, so only ``k`` needs that precondition.
    By default, lowering arranges scratch; this planner records its contract.
    """

    specialization = make_block_topk_specialization(
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
    requirements = _build_execution_requirements(
        resolved,
        launch,
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
        implementation=specialization,
        topology=requirements.topology,
        participation=requirements.participation,
        result=result,
        synchronization=requirements.synchronization,
        temp_storage=requirements.temp_storage,
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_topk.cuh",
            cpp_class="cub::detail::block_topk",
            method=specialization.method_name,
        ),
    )


_register_group_operation_family(
    GroupTopKSemantics,
    classifications=_classifications,
    planner=_plan_topk,
    group_kinds=frozenset({"block"}),
    unsupported_group_message="TopK supports only complete this_block() groups",
)
