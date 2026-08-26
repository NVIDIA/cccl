# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402, F401

"""Shared constructors for the initial group-planner contract tests."""

from cuda.coop._core import (
    COMPLETE_WARP_GROUP_KINDS,
    THREAD_GROUP_KINDS,
    AlgorithmSpec,
    ArgumentBinding,
    ArgumentKind,
    CudaxReturnKind,
    CxxFunction,
    CxxOperator,
    Dependency,
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupReduceSemantics,
    GroupScanMode,
    GroupScanSemantics,
    LaunchFactOrigin,
    LaunchFacts,
    LogicalResultContract,
    ParameterRole,
    PreconditionEnforcement,
    ResultOwnership,
    ResultVisibility,
    RuntimeValue,
    StatefulOperator,
    StorageOwnership,
    SynchronizationScope,
    ThreadGroup,
    ThreadHierarchy,
    UnsupportedReasonCode,
    make_group_primitive_call,
    make_scan_semantics,
    merge_launch_facts,
    plan_group_primitive,
    resolve_thread_group,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)
from cuda.coop._core.block import (
    BlockReduceAlgorithm,
    BlockScanAlgorithm,
    make_block_reduce_semantics,
    make_block_scan_spec,
)
from cuda.coop._core.warp import make_warp_reduce_spec, make_warp_scan_spec


def _reduce(**overrides):
    broadcast = overrides.pop("broadcast", True)
    cub_algorithm = overrides.pop("cub_algorithm", None)
    dtype = overrides.pop("dtype", "int")
    operation = overrides.pop("operation", "sum")
    operand_kind = GroupOperandKind(overrides.pop("operand_kind", "scalar"))
    reduce_operator = overrides.pop("reduce_operator", None)
    if operation == "max":
        operation = "reduce"
        reduce_operator = CxxOperator("::cuda::maximum<>", dtype)
    primitive = make_block_reduce_semantics(
        dtype=dtype,
        operation=operation,
        value_kind=operand_kind.value,
        items_per_thread=overrides.pop("items_per_thread", 1),
        valid_items=overrides.pop("valid_items", False),
        reduce_operator=reduce_operator,
    )
    assert not overrides
    return GroupReduceSemantics(
        primitive=primitive,
        broadcast=broadcast,
        cub_algorithm=cub_algorithm,
    )


def _scan(**overrides):
    cub_algorithm = overrides.pop("cub_algorithm", None)
    valid_items = overrides.pop("valid_items", ArgumentBinding.omitted())
    operand_kind = GroupOperandKind(overrides.pop("operand_kind", "scalar"))
    primitive = make_scan_semantics(
        dtype=overrides.pop("dtype", "int"),
        mode=overrides.pop("mode", "exclusive"),
        value_kind=operand_kind.value,
        items_per_thread=overrides.pop("items_per_thread", 1),
        scan_operator=overrides.pop("scan_operator", None),
        initial_value=overrides.pop("initial_value", None),
        aggregate=overrides.pop("aggregate", False),
        prefix_callback=overrides.pop("prefix_callback", None),
    )
    assert not overrides
    return GroupScanSemantics(
        primitive,
        cub_algorithm=cub_algorithm,
        valid_items=valid_items,
    )


def _load_store(kind="load", **overrides):
    operation = GroupLoadStoreSemantics(
        kind=GroupLoadStoreKind(kind),
        dtype=overrides.pop("dtype", "int"),
        items_per_thread=overrides.pop("items_per_thread", 2),
        algorithm=overrides.pop("algorithm", GroupLoadStoreAlgorithm.DIRECT),
        valid_items=overrides.pop("valid_items", ArgumentBinding.omitted()),
        oob_default=overrides.pop("oob_default", ArgumentBinding.omitted()),
        offset=overrides.pop("offset", ArgumentBinding.omitted()),
    )
    assert not overrides
    return operation


def _plan(group, operation, launch=(64, 1, 1)):
    facts = launch if isinstance(launch, LaunchFacts) else LaunchFacts(launch)
    return plan_group_primitive(
        make_group_primitive_call(group, operation),
        facts,
    )


_COMPLETE_WARP_GROUP_SAMPLES = (
    this_warp(),
    this_warp().group_by(8),
    this_block().group_by(1, exhaustive=False),
)
_NON_COMPLETE_WARP_GROUP_SAMPLES = (
    this_thread(),
    this_block(),
    this_cluster(),
    this_grid(),
)
