# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared constructors for Block Load/Store planner contract tests."""

from cuda.coop._core import (
    ArgumentBinding,
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    GroupOperandKind,
    GroupScanSemantics,
    LaunchFacts,
    StorageOwnership,
    make_group_primitive_call,
    make_scan_semantics,
    plan_group_primitive,
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
    )
    assert not overrides
    return GroupScanSemantics(
        primitive,
        cub_algorithm=cub_algorithm,
        valid_items=valid_items,
    )


def _load_store(kind="load", **overrides):
    algorithm = GroupLoadStoreAlgorithm(
        overrides.pop("algorithm", GroupLoadStoreAlgorithm.DIRECT)
    )
    default_storage_ownership = StorageOwnership.IMPLEMENTATION
    operation = GroupLoadStoreSemantics(
        kind=GroupLoadStoreKind(kind),
        dtype=overrides.pop("dtype", "int"),
        items_per_thread=overrides.pop("items_per_thread", 2),
        algorithm=algorithm,
        valid_items=overrides.pop("valid_items", ArgumentBinding.omitted()),
        oob_default=overrides.pop("oob_default", ArgumentBinding.omitted()),
        offset=overrides.pop("offset", ArgumentBinding.omitted()),
        storage_ownership=overrides.pop("storage_ownership", default_storage_ownership),
        storage_sharing=overrides.pop("storage_sharing", None),
        storage_size_in_bytes=overrides.pop("storage_size_in_bytes", None),
        storage_alignment=overrides.pop("storage_alignment", None),
        storage_auto_sync=overrides.pop(
            "storage_auto_sync",
            True,
        ),
    )
    assert not overrides
    return operation


def _plan(group, operation, launch=(64, 1, 1)):
    facts = launch if isinstance(launch, LaunchFacts) else LaunchFacts(launch)
    return plan_group_primitive(
        make_group_primitive_call(group, operation),
        facts,
    )
