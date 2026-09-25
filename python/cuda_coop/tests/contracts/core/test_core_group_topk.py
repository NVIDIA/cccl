# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TopK participation contracts remain independent of compiler packages."""

import pytest

from cuda.coop._core import (
    INT32,
    ArgumentBinding,
    LaunchFacts,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
)
from cuda.coop._core.group.topk import GroupTopKSemantics


@pytest.mark.parametrize(
    "k",
    [ArgumentBinding.static(3), ArgumentBinding.runtime()],
    ids=["static-k", "runtime-k"],
)
@pytest.mark.parametrize(
    "valid_items, expected_uniform",
    [
        (ArgumentBinding.omitted(), ("k",)),
        (ArgumentBinding.static(8), ("k", "valid_items")),
        (ArgumentBinding.runtime(), ("k", "valid_items")),
    ],
    ids=["full-tile", "static-valid-items", "runtime-valid-items"],
)
def test_topk_plan_records_uniform_counts(k, valid_items, expected_uniform):
    operation = GroupTopKSemantics(
        key_dtype=INT32,
        items_per_thread=2,
        selection="min",
        k=k,
        valid_items=valid_items,
    )
    plan = plan_group_primitive(
        make_group_primitive_call(this_block(), operation), LaunchFacts(64)
    ).require_supported()

    assert plan.participation.uniform_arguments == expected_uniform
