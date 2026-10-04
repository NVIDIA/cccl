# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build Load, Store, and Scan requests shared by planner tests.

Tests can vary one option while keeping the element type, item count, and
launch dimensions fixed. Planning these requests needs no kernel compiler.
"""

from cuda.coop._core import (
    ArgumentBinding,
    GroupLoadStoreSemantics,
    GroupOperandKind,
    GroupScanSemantics,
    LaunchFacts,
    make_group_primitive_call,
    make_scan_semantics,
    plan_group_primitive,
)


def _load_store(kind="load", **overrides):
    options = {"dtype": "int", "items_per_thread": 2, "algorithm": "direct"}
    options.update(overrides)
    return GroupLoadStoreSemantics(kind=kind, **options)


def _plan(group, operation, launch=(64, 1, 1)):
    facts = launch if isinstance(launch, LaunchFacts) else LaunchFacts(launch)
    return plan_group_primitive(
        make_group_primitive_call(group, operation), facts
    )


def _scan(**overrides):
    """Build a scan request with separate algorithm and group controls.

    Start with a scalar exclusive sum, then apply each test's overrides.
    The primitive describes input shape and scan behavior. The group wrapper
    adds the CUB algorithm and valid-prefix binding. Reject unused overrides
    so a misspelled option cannot silently leave the default in place.
    """

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
