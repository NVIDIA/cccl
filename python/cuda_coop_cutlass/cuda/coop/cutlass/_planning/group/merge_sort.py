# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Select a Merge Sort implementation and describe the group result.

A shared call record supplies payloads and the comparator. The group plan
adds exact launch dimensions, static prefix validation, result ownership,
and scratch reuse rules. The group API treats inputs as read-only even
though the selected CUB primitive mutates arrays; the backend allocates and
sorts copies to implement the returned payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._bindings import ArgumentBinding, BindingKind, normalize_i32_binding
from .._types import ArgumentKind, ParameterClassification, ParameterRole
from ..block.merge_sort import (
    BlockMergeSortSemantics,
    make_block_merge_sort_specialization,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ..warp.merge_sort import make_warp_merge_sort_specialization
from ._dispatch import _register_group_operation_family
from ._execution_requirements import (
    _build_execution_requirements,
    _unsupported,
    _unsupported_cub_warp_width,
)
from ._model import (
    ArgumentPrecondition,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupPrimitiveCall,
    ImplementationProvenance,
    LogicalResultContract,
    PreconditionEnforcement,
    ResultContract,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    UnsupportedReasonCode,
)


@dataclass(frozen=True, eq=False)
class GroupMergeSortSemantics:
    """Add valid-prefix binding information to a shape-independent call.

    ``primitive`` is the shared BlockMergeSortSemantics record, also used
    when planning a Warp call. Its tile policy must be partial exactly when
    valid_items is bound, either statically or at run time. A static count
    must fit in a signed 32-bit integer and is stored as a Python int; runtime
    bindings pass through unchanged. The planner later checks static counts
    against the tile size.
    """

    primitive: BlockMergeSortSemantics
    valid_items: ArgumentBinding = field(
        default_factory=ArgumentBinding.omitted
    )

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockMergeSortSemantics):
            raise TypeError("primitive must be BlockMergeSortSemantics")
        if not isinstance(self.valid_items, ArgumentBinding):
            raise TypeError("valid_items must be an ArgumentBinding")
        object.__setattr__(
            self,
            "valid_items",
            normalize_i32_binding(self.valid_items, name="valid_items"),
        )
        if self.primitive.has_partial_tile != (
            self.valid_items.kind is not BindingKind.OMITTED
        ):
            raise ValueError(
                "partial Merge Sort requires a valid_items binding"
            )

    @property
    def dtype(self) -> Any:
        return self.primitive.key_dtype

    @property
    def result_visibility(self) -> ResultVisibility:
        return ResultVisibility.PER_MEMBER

    @property
    def returns_value(self) -> bool:
        return True

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key, self.valid_items.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupMergeSortSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _classifications(operation):
    """Describe public operands independently of CUB's inout arrays.

    Keys and values are inputs at the group API. The backend later copies
    them for CUB. The comparator is static, while a partial call also carries
    its count binding and runtime padding key. This classification supports
    shared planning and diagnostics without choosing a wrapper signature.
    """

    result = [
        ParameterClassification(
            "keys", ArgumentKind.RUNTIME, ParameterRole.INPUT
        )
    ]
    if operation.primitive.has_values:
        result.append(
            ParameterClassification(
                "values", ArgumentKind.RUNTIME, ParameterRole.INPUT
            )
        )
    result.append(
        ParameterClassification(
            "compare_op", ArgumentKind.STATIC, ParameterRole.OPERATOR
        )
    )
    if operation.primitive.has_partial_tile:
        result.extend(
            (
                ParameterClassification(
                    "valid_items",
                    operation.valid_items.argument_kind,
                    ParameterRole.INPUT,
                ),
                ParameterClassification(
                    "oob_default", ArgumentKind.RUNTIME, ParameterRole.INPUT
                ),
            )
        )
    return tuple(result)


def _plan_merge_sort(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupMergeSortSemantics,
) -> GroupLoweringPlan:
    """Choose a block or Warp Sort and record tile and result contracts.

    The group width times items_per_thread sets the tile capacity. Exact
    launch dimensions select the block specialization and participation.
    Check static counts now and retain runtime bounds and uniformity
    requirements in the plan. Each member receives a keys payload and, for
    pairs, a corresponding values payload.

    Parameters
    ----------
    call : GroupPrimitiveCall
        Group and Merge Sort semantics requested by the frontend.
    resolved : ThreadGroup
        Group resolved against the launch dimensions by shared dispatch.
    launch : LaunchFacts
        Exact block shape needed for participation and specialization.
    operation : GroupMergeSortSemantics
        Payload, comparator, and valid-prefix binding for this call.

    Returns
    -------
    GroupLoweringPlan
        CUB implementation with per-member results and implementation-owned
        scratch, or an unsupported plan for a block/warp shape restriction.
        The backend resolves scratch size and emits the reuse barrier.

    Raises
    ------
    ValueError
        A static valid count is outside zero through the tile capacity, or
        primitive factory arguments fail validation.

    Notes
    -----
    Partial-tile factories select their overload from the presence of both
    controls. The zero placeholders below select that overload; they are not
    replacements for the caller's count or padding key. Runtime checking is
    performed by the checked C++ provider selected by those factories.
    """

    primitive = operation.primitive
    kwargs = {
        "key_dtype": primitive.key_dtype,
        "value_dtype": primitive.value_dtype,
        "items_per_thread": primitive.items_per_thread,
        "compare_operator": primitive.compare_operator,
        "valid_items": 0 if primitive.has_partial_tile else None,
        "oob_default": 0 if primitive.has_partial_tile else None,
    }
    assert launch.exact_block_dim is not None
    if resolved.kind == "block":
        width = launch.exact_block_threads
        assert width is not None
        if width & (width - 1):
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "cub::BlockMergeSort requires "
                "a power-of-two block thread count",
            )
        specialization = make_block_merge_sort_specialization(
            block_dim=launch.exact_block_dim, **kwargs
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        header = "cub/block/block_merge_sort.cuh"
        cpp_class = "cub::BlockMergeSort"
    else:
        width, error = _unsupported_cub_warp_width(call, resolved)
        if error is not None:
            return error
        assert width is not None
        specialization = make_warp_merge_sort_specialization(
            threads_in_warp=width, **kwargs
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        header = "cub/warp/warp_merge_sort.cuh"
        cpp_class = "cub::WarpMergeSort"
    capacity = width * primitive.items_per_thread
    if (
        operation.valid_items.kind is BindingKind.STATIC
        and not 0 <= operation.valid_items.value <= capacity
    ):
        raise ValueError(
            "valid_items must be between 0 and the group tile size "
            f"({capacity})"
        )
    outputs = [("keys", primitive.key_dtype)]
    if primitive.has_values:
        outputs.append(("values", primitive.value_dtype))
    result = ResultContract(
        tuple(
            LogicalResultContract(
                name=name,
                dtype=dtype,
                visibility=ResultVisibility.PER_MEMBER,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.ARRAY,
                items_per_member=primitive.items_per_thread,
            )
            for name, dtype in outputs
        )
    )
    requirements = _build_execution_requirements(
        resolved,
        launch,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=("valid_items", "oob_default")
        if primitive.has_partial_tile
        else (),
        argument_preconditions=(
            ArgumentPrecondition(
                name="valid_items",
                minimum=0,
                maximum=capacity,
                enforcement=(
                    PreconditionEnforcement.PLANNER_VALIDATED
                    if operation.valid_items.kind is BindingKind.STATIC
                    else PreconditionEnforcement.CALLER
                ),
            ),
        )
        if primitive.has_partial_tile
        else (),
    )
    return GroupLoweringPlan(
        target=target,
        call=call,
        resolved_group=resolved,
        implementation=specialization,
        topology=requirements.topology,
        participation=requirements.participation,
        result=result,
        synchronization=requirements.synchronization,
        temp_storage=requirements.temp_storage,
        provenance=ImplementationProvenance(
            library="CUB", header=header, cpp_class=cpp_class, method="Sort"
        ),
    )


_register_group_operation_family(
    GroupMergeSortSemantics,
    classifications=_classifications,
    planner=_plan_merge_sort,
    group_kinds=frozenset({"block", "warp", "threads_within_warp"}),
    unsupported_group_message=(
        "Merge Sort supports complete block, physical warp, "
        "and power-of-two logical warp groups"
    ),
)

__all__ = [
    "GroupMergeSortSemantics",
]
