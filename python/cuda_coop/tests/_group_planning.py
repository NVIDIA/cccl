# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Convenience constructors for Load/Store planner tests."""

from cuda.coop._core import (
    GroupLoadStoreSemantics,
    LaunchFacts,
    make_group_primitive_call,
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
