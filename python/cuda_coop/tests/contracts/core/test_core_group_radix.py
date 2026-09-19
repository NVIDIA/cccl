# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Radix group contracts remain independent of compiler packages."""

import pytest

from cuda.coop._core import (
    INT32,
    UINT64,
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
    this_warp,
)
from cuda.coop._core.block.radix_rank import make_block_radix_rank_semantics
from cuda.coop._core.block.radix_sort import make_block_radix_sort_semantics
from cuda.coop._core.group.radix import GroupRadixRankSemantics, GroupRadixSortSemantics


@pytest.mark.parametrize("pairs", [False, True])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("striped", [False, True])
def test_sort_plan_describes_cub_method_results_and_storage(pairs, descending, striped):
    operation = GroupRadixSortSemantics(
        make_block_radix_sort_semantics(
            key_dtype=UINT64,
            value_dtype=INT32 if pairs else None,
            items_per_thread=3,
            descending=descending,
            blocked_to_striped=striped,
        )
    )
    plan = plan_group_primitive(
        make_group_primitive_call(this_block(), operation), LaunchFacts((16, 4, 1))
    ).require_supported()
    method = "SortDescending" if descending else "Sort"
    if striped:
        method += "BlockedToStriped"
    assert plan.provenance.cpp_class == "cub::BlockRadixSort"
    assert plan.provenance.method == method
    assert tuple(value.dtype for value in plan.result.values) == (
        (UINT64, INT32) if pairs else (UINT64,)
    )
    assert all(value.items_per_member == 3 for value in plan.result.values)
    assert plan.participation.exact_block_dim == (16, 4, 1)
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK


def test_rank_plan_has_fixed_result_dtype_and_rejects_warp_scope():
    operation = GroupRadixRankSemantics(
        make_block_radix_rank_semantics(
            key_dtype=UINT64,
            items_per_thread=3,
            begin_bit=60,
            end_bit=64,
            key_bit_width=64,
        )
    )
    plan = plan_group_primitive(
        make_group_primitive_call(this_block(), operation), LaunchFacts(64)
    ).require_supported()
    assert plan.result.values[0].dtype is INT32
    assert plan.result.values[0].items_per_member == 3
    assert plan.provenance.cpp_class == "cub::BlockRadixRank"
    unsupported = plan_group_primitive(
        make_group_primitive_call(this_warp(), operation), LaunchFacts(64)
    )
    assert unsupported.unsupported is not None


@pytest.mark.parametrize(
    "operation,options",
    [
        ("radix_sort_keys", {"descending": 1}),
        ("radix_sort_keys", {"begin_bit": -1}),
        ("radix_sort_keys", {"end_bit": 33}),
        ("radix_rank", {"radix_bits": 9}),
        ("radix_rank", {"begin_bit": 31}),
        ("radix_rank", {"end_bit": 6, "radix_bits": 4}),
    ],
)
def test_common_frontend_rejects_invalid_controls_before_dispatch(operation, options):
    from cuda.coop._core.api import _dispatch, radix

    class Payload:
        items_per_thread = 2
        dtype = int

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

    with _dispatch._compiler_scope("test.backend"):
        with pytest.raises((TypeError, ValueError)):
            getattr(radix, operation)(this_block(), Payload(), **options)
