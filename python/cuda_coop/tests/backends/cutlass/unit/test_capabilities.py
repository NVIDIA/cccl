# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import ast
import itertools
import os
import subprocess
import sys
import textwrap
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT

from cuda.coop._core import (
    ArgumentBinding,
    CxxOperator,
    GroupExchangeSemantics,
    GroupLoweringTarget,
    GroupReduceSemantics,
    GroupScanSemantics,
    LaunchFacts,
    make_group_primitive_call,
    make_reduce_semantics,
    make_scan_semantics,
    plan_group_primitive,
    this_block,
    this_warp,
)
from cuda.coop._core.block import BlockReduceAlgorithm, make_block_exchange_semantics
from cuda.coop.cutlass._capabilities import (
    BLOCK_ALIAS_BINDINGS,
    BLOCK_BINDING_BY_NAME,
    BLOCK_EXPORT_BINDINGS,
    BLOCK_FACTORY_BINDINGS,
    BLOCK_OPERATION_BINDINGS,
    BLOCK_SUPPORT_BINDINGS,
    CAPABILITIES,
    CAPABILITY_BY_KEY,
    GROUP_METHOD_CAPABILITIES,
    GROUP_METHOD_CAPABILITY_BY_KIND,
    WARP_ALIAS_BINDINGS,
    WARP_BINDING_BY_NAME,
    WARP_EXPORT_BINDINGS,
    WARP_FACTORY_BINDINGS,
    WARP_OPERATION_BINDINGS,
    WARP_SUPPORT_BINDINGS,
    ApiAvailability,
    ApiStability,
    ExportBindingKind,
    GroupFirstReadiness,
    GroupFirstStage,
    GroupKind,
    OperandKind,
    OperationFamily,
    ProvenanceKind,
    binding_for,
    capability_for,
    group_first_planner_models_binding,
    group_method_capability_for,
    resolved_binding_selectors,
)
from cuda.coop.cutlass._limits import MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD


def _literal_all(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, list)
            assert all(isinstance(name, str) for name in value)
            return value
    raise AssertionError(f"{path} has no literal __all__ manifest")


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"{path} has no function {name}")


def _assert_export_partition(
    *,
    group: GroupKind,
    bindings,
    binding_by_name,
    source_path: Path,
) -> None:
    public_names = _literal_all(source_path)
    binding_names = [binding.name for binding in bindings]

    assert len(binding_names) == len(set(binding_names))
    assert set(binding_names) == set(public_names)
    assert set(binding_by_name) == set(public_names)
    assert all(binding_by_name[name].name == name for name in public_names)

    for binding in bindings:
        if binding.kind is ExportBindingKind.SUPPORT:
            assert binding.family is None
            continue
        assert binding.family is not None
        assert capability_for(binding.family, group).group is group


def test_block_export_bindings_partition_the_current_public_surface():
    _assert_export_partition(
        group=GroupKind.BLOCK,
        bindings=BLOCK_EXPORT_BINDINGS,
        binding_by_name=BLOCK_BINDING_BY_NAME,
        source_path=(
            SOURCE_ROOT / "cuda" / "coop" / "cutlass" / "_dsl" / "block" / "__init__.py"
        ),
    )
    assert len(BLOCK_EXPORT_BINDINGS) == 77
    assert len(BLOCK_OPERATION_BINDINGS) == 20
    assert len(BLOCK_ALIAS_BINDINGS) == 24
    assert len(BLOCK_FACTORY_BINDINGS) == 26
    assert len(BLOCK_SUPPORT_BINDINGS) == 6


def test_warp_export_bindings_partition_the_current_public_surface():
    _assert_export_partition(
        group=GroupKind.WARP,
        bindings=WARP_EXPORT_BINDINGS,
        binding_by_name=WARP_BINDING_BY_NAME,
        source_path=(
            SOURCE_ROOT / "cuda" / "coop" / "cutlass" / "_dsl" / "warp" / "__init__.py"
        ),
    )
    assert len(WARP_EXPORT_BINDINGS) == 32
    assert len(WARP_OPERATION_BINDINGS) == 7
    assert len(WARP_ALIAS_BINDINGS) == 10
    assert len(WARP_FACTORY_BINDINGS) == 13
    assert len(WARP_SUPPORT_BINDINGS) == 2


def test_aliases_factories_and_stateful_adapters_name_real_targets():
    for group, bindings in (
        (GroupKind.BLOCK, BLOCK_BINDING_BY_NAME),
        (GroupKind.WARP, WARP_BINDING_BY_NAME),
    ):
        for binding in bindings.values():
            resolved_binding_selectors(group, binding.name)
            if binding.kind not in {
                ExportBindingKind.ALIAS,
                ExportBindingKind.FACTORY,
                ExportBindingKind.STATEFUL_ADAPTER,
            }:
                continue
            assert binding.target_export is not None
            target = bindings[binding.target_export]
            assert target.kind is not ExportBindingKind.SUPPORT
            assert target.family is binding.family

    assert dict(
        binding_for(GroupKind.BLOCK, "exchange_blocked_to_striped").selectors
    ) == {"mode": "blocked_to_striped"}
    assert dict(binding_for("warp", "exclusive_sum").selectors) == {
        "scan_mode": "exclusive",
        "operator": "sum",
    }
    assert binding_for("block", "make_run_length").target_export == "run_length"
    assert binding_for("block", "run_length").target_export == "run_length_decode"
    assert dict(resolved_binding_selectors("warp", "make_exclusive_sum")) == {
        "scan_mode": "exclusive",
        "operator": "sum",
    }


@pytest.mark.parametrize("lookup", [binding_for, resolved_binding_selectors])
def test_scoped_binding_lookup_rejects_non_scoped_groups(lookup):
    with pytest.raises(KeyError, match="'thread' has no scoped export bindings"):
        lookup(GroupKind.THREAD, "reduce")


def test_capability_keys_are_complete_and_unique_for_bound_operations():
    assert len(CAPABILITIES) == len(CAPABILITY_BY_KEY)
    assert all(
        CAPABILITY_BY_KEY[capability.family, capability.group] is capability
        for capability in CAPABILITIES
    )

    expected_keys = {
        (binding.family, group)
        for group, bindings in (
            (GroupKind.BLOCK, BLOCK_EXPORT_BINDINGS),
            (GroupKind.WARP, WARP_EXPORT_BINDINGS),
        )
        for binding in bindings
        if binding.family is not None
    }
    expected_keys.update(
        {
            (OperationFamily.REDUCE, GroupKind.THREAD),
            (OperationFamily.REDUCE, GroupKind.THREADS_WITHIN_WARP),
            (OperationFamily.REDUCE, GroupKind.WARPS_WITHIN_BLOCK),
            (OperationFamily.REDUCE, GroupKind.CLUSTER),
            (OperationFamily.REDUCE, GroupKind.GRID),
            (OperationFamily.MERGE_SORT, GroupKind.THREADS_WITHIN_WARP),
        }
    )
    assert set(CAPABILITY_BY_KEY) == expected_keys

    with pytest.raises(ValueError, match="not a valid OperationFamily"):
        capability_for("unknown", "block")
    with pytest.raises(KeyError, match="unknown"):
        binding_for("block", "unknown")
    with pytest.raises(KeyError, match="no scoped export bindings"):
        binding_for("grid", "reduce")


def test_group_method_inventory_covers_every_static_group_form():
    assert len(GROUP_METHOD_CAPABILITIES) == len(GroupKind)
    assert set(GROUP_METHOD_CAPABILITY_BY_KIND) == set(GroupKind)
    for group in GroupKind:
        capability = group_method_capability_for(group)
        assert capability.group is group
        assert capability.query_levels == (
            "thread",
            "warp",
            "block",
            "cluster",
            "grid",
        )
        assert capability.membership
        assert capability.synchronization

    assert group_method_capability_for("block").readiness is GroupFirstReadiness.READY
    assert group_method_capability_for("warp").readiness is GroupFirstReadiness.READY
    assert all(
        capability.readiness is GroupFirstReadiness.READY
        and capability.validation_evidence
        for capability in GROUP_METHOD_CAPABILITIES
    )


def test_group_first_readiness_and_target_invariants_are_explicit():
    for capability in CAPABILITIES:
        assert isinstance(capability.planned_target, GroupLoweringTarget)
        if capability.primary_group_first_readiness is GroupFirstReadiness.READY:
            assert capability.unsupported_reason is None
            assert capability.group_first_provenance is not None
        else:
            assert capability.unsupported_reason
            if (
                capability.primary_group_first_readiness
                is GroupFirstReadiness.BLOCKED_PROVIDER_PARITY
            ):
                assert capability.group_first_provenance is not None
            else:
                assert capability.group_first_provenance is None

        if capability.planned_target is GroupLoweringTarget.UNSUPPORTED:
            assert (
                capability.primary_group_first_readiness
                is GroupFirstReadiness.UNSUPPORTED
            )
            assert capability.planned_api is None
        else:
            assert (
                capability.primary_group_first_readiness
                is not GroupFirstReadiness.UNSUPPORTED
            )
            assert capability.planned_api is not None
            assert capability.planned_api.headers
            assert capability.planned_api.entity
            assert capability.planned_api.methods

        planner_is_ready = capability.primary_group_first_readiness in {
            GroupFirstReadiness.READY,
            GroupFirstReadiness.READY_NOT_EXPOSED,
            GroupFirstReadiness.BLOCKED_PROVIDER_PARITY,
            GroupFirstReadiness.BLOCKED_PROVIDER_CONVERSION,
        }
        assert bool(capability.group_first_operand_forms) is planner_is_ready

        if (
            capability.primary_group_first_readiness is GroupFirstReadiness.READY
            and capability.group_first_readiness
            is GroupFirstReadiness.BLOCKED_PROVIDER_PARITY
        ):
            assert any(
                route.readiness is GroupFirstReadiness.BLOCKED_PROVIDER_PARITY
                and route.unsupported_reason
                for route in capability.alternate_group_first_routes
            )

    assert capability_for("reduce", "block").remaining_group_first_stages == ()
    for group in ("block", "warp"):
        assert capability_for("scan", group).remaining_group_first_stages == ()
    assert capability_for("row_reduce", "block").remaining_group_first_stages == (
        GroupFirstStage.DEPENDENCY,
        GroupFirstStage.PLANNER,
        GroupFirstStage.ROOT_EXPOSURE,
        GroupFirstStage.PARITY_VALIDATION,
    )


@pytest.mark.parametrize(
    ("family", "group"),
    [
        (OperationFamily.LOAD, GroupKind.BLOCK),
        (OperationFamily.LOAD, GroupKind.WARP),
        (OperationFamily.STORE, GroupKind.BLOCK),
        (OperationFamily.STORE, GroupKind.WARP),
        (OperationFamily.SCAN, GroupKind.BLOCK),
        (OperationFamily.SCAN, GroupKind.WARP),
        (OperationFamily.EXCHANGE, GroupKind.BLOCK),
        (OperationFamily.EXCHANGE, GroupKind.WARP),
        (OperationFamily.ADJACENT_DIFFERENCE, GroupKind.BLOCK),
        (OperationFamily.DISCONTINUITY, GroupKind.BLOCK),
        (OperationFamily.HISTOGRAM, GroupKind.BLOCK),
        (OperationFamily.RUN_LENGTH_DECODE, GroupKind.BLOCK),
        (OperationFamily.SHUFFLE, GroupKind.BLOCK),
        (OperationFamily.MERGE_SORT, GroupKind.BLOCK),
        (OperationFamily.MERGE_SORT, GroupKind.WARP),
        (OperationFamily.RADIX_RANK, GroupKind.BLOCK),
        (OperationFamily.RADIX_SORT, GroupKind.BLOCK),
    ],
)
def test_public_cub_capabilities_are_ready(family, group):
    capability = capability_for(family, group)

    assert capability.group_first_readiness is GroupFirstReadiness.READY
    assert capability.unsupported_reason is None
    assert capability.remaining_group_first_stages == ()


def test_capability_readiness_inventory_matches_implementations():
    assert Counter(
        capability.group_first_readiness for capability in CAPABILITIES
    ) == Counter(
        {
            GroupFirstReadiness.READY: 25,
            GroupFirstReadiness.BLOCKED_PROVIDER_CONVERSION: 1,
            GroupFirstReadiness.BLOCKED_PLANNER_AND_DEPENDENCY: 1,
        }
    )
    assert {
        (capability.family, capability.group)
        for capability in CAPABILITIES
        if capability.group_first_readiness
        is GroupFirstReadiness.BLOCKED_PROVIDER_PARITY
    } == set()

    logical_merge_sort = capability_for(
        OperationFamily.MERGE_SORT, GroupKind.THREADS_WITHIN_WARP
    )
    assert logical_merge_sort.group_first_readiness is GroupFirstReadiness.READY

    for group in (
        GroupKind.THREAD,
        GroupKind.THREADS_WITHIN_WARP,
        GroupKind.WARPS_WITHIN_BLOCK,
        GroupKind.CLUSTER,
    ):
        assert (
            capability_for(OperationFamily.REDUCE, group).group_first_readiness
            is GroupFirstReadiness.READY
        )
    grid_reduce = capability_for(OperationFamily.REDUCE, GroupKind.GRID)
    assert (
        grid_reduce.group_first_readiness
        is GroupFirstReadiness.BLOCKED_PROVIDER_CONVERSION
    )
    assert grid_reduce.remaining_group_first_stages == (
        GroupFirstStage.PROVIDER,
        GroupFirstStage.ROOT_EXPOSURE,
        GroupFirstStage.PARITY_VALIDATION,
    )
    assert (
        grid_reduce.unsupported_reason
        == "grid Reduce is blocked until the CUTLASS DSL provides a reviewed "
        "compiler-managed device workspace contract"
    )


def test_reduce_scan_and_exchange_statuses_match_current_implementations():
    for group in (GroupKind.BLOCK, GroupKind.WARP):
        reduce = capability_for(OperationFamily.REDUCE, group)
        assert reduce.planned_target is GroupLoweringTarget.CUDAX_GROUP
        assert reduce.primary_group_first_readiness is GroupFirstReadiness.READY
        assert reduce.group_first_readiness is GroupFirstReadiness.READY
        assert reduce.group_first_provenance is not None
        assert reduce.group_first_provenance.kind is ProvenanceKind.CUDAX_PUBLIC
        assert reduce.planned_api is not None
        assert reduce.planned_api.entity == "::cuda::experimental::coop"
        assert reduce.planned_api.methods == ("reduce",)

    block_reduce = capability_for("reduce", "block")
    assert [
        selector.name for selector in block_reduce.group_first_selector_support
    ] == [
        "broadcast",
        "valid_items",
        "algorithm",
    ]
    assert len(block_reduce.alternate_group_first_routes) == 1
    block_route = block_reduce.alternate_group_first_routes[0]
    assert block_route.target is GroupLoweringTarget.CUB_BLOCK
    assert block_route.api.entity == "::cub::BlockReduce"
    assert block_route.readiness is GroupFirstReadiness.READY

    warp_reduce = capability_for("reduce", "warp")
    assert [selector.name for selector in warp_reduce.group_first_selector_support] == [
        "broadcast",
        "valid_items",
    ]
    assert len(warp_reduce.alternate_group_first_routes) == 1
    warp_route = warp_reduce.alternate_group_first_routes[0]
    assert warp_route.target is GroupLoweringTarget.CUB_WARP
    assert warp_route.api.entity == "::cub::WarpReduce"
    assert warp_route.readiness is GroupFirstReadiness.READY

    for reduce in (block_reduce, warp_reduce):
        assert reduce.builtin_scoped_provenance.kind is ProvenanceKind.CUDAX_PUBLIC
        assert reduce.builtin_scoped_provenance.api is reduce.planned_api

    assert [
        selector.name for selector in block_reduce.builtin_scoped_selector_support
    ] == ["valid_items", "algorithm"]
    assert len(block_reduce.alternate_builtin_scoped_routes) == 1
    assert (
        block_reduce.alternate_builtin_scoped_routes[0].target
        is GroupLoweringTarget.CUB_BLOCK
    )
    assert [
        selector.name for selector in warp_reduce.builtin_scoped_selector_support
    ] == ["threads_in_warp", "valid_items"]
    assert len(warp_reduce.alternate_builtin_scoped_routes) == 2
    assert all(
        route.target is GroupLoweringTarget.CUB_WARP
        for route in warp_reduce.alternate_builtin_scoped_routes
    )
    assert [
        route.provenance.kind for route in warp_reduce.alternate_builtin_scoped_routes
    ] == [
        ProvenanceKind.CUB_PUBLIC,
        ProvenanceKind.CUB_PUBLIC_WITH_GENERATED_ADAPTER,
    ]

    block_scan = capability_for("scan", "block")
    assert block_scan.planned_target is GroupLoweringTarget.CUB_BLOCK
    assert block_scan.group_first_readiness is GroupFirstReadiness.READY
    assert block_scan.planned_api is not None
    assert block_scan.planned_api.entity == "::cub::BlockScan"
    assert block_scan.group_first_provenance is block_scan.builtin_scoped_provenance
    assert block_scan.group_first_provenance.kind is ProvenanceKind.CUB_PUBLIC
    assert [
        (selector.name, selector.accepted_values)
        for selector in block_scan.group_first_selector_support
    ] == [
        ("algorithm", ("raking", "raking_memoize", "warp_scans")),
    ]
    with pytest.raises(ValueError, match="materialized group-first providers"):
        replace(block_scan, group_first_provenance=None)

    block_exchange = capability_for("exchange", "block")
    assert block_exchange.planned_target is GroupLoweringTarget.CUB_BLOCK
    assert block_exchange.group_first_readiness is GroupFirstReadiness.READY
    assert block_exchange.builtin_scoped_provenance.kind is ProvenanceKind.CUB_PUBLIC
    assert block_exchange.builtin_scoped_provenance.api == block_exchange.planned_api
    assert block_exchange.group_first_provenance.kind is ProvenanceKind.CUB_PUBLIC
    assert (
        block_exchange.group_first_operand_forms[0].max_items_per_thread
        == MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD
    )
    assert "logical/scatter WarpExchange compatibility routes retain x4" in (
        block_exchange.builtin_scoped_operand_forms[0].note
    )
    assert block_exchange.planned_api is not None
    assert block_exchange.planned_api.entity == "::cub::BlockExchange"
    assert block_exchange.planned_api.methods == (
        "StripedToBlocked",
        "BlockedToStriped",
    )
    assert group_first_planner_models_binding("block", "exchange_blocked_to_striped")
    assert not group_first_planner_models_binding(
        "block", "exchange_scatter_to_blocked"
    )
    assert [
        selector.name for selector in block_exchange.builtin_scoped_selector_support
    ] == ["mode"]
    assert len(block_exchange.alternate_builtin_scoped_routes) == 1
    block_compatibility = block_exchange.alternate_builtin_scoped_routes[0]
    assert block_compatibility.api.methods[-1] == "ScatterToStripedFlagged"
    assert block_compatibility.matches(
        OperandKind.THREAD_DATA,
        {"mode": "scatter_to_blocked"},
        items_per_thread=MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD,
    )
    assert not block_compatibility.matches(
        OperandKind.THREAD_DATA,
        {"mode": "scatter_to_blocked"},
        items_per_thread=MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD + 1,
    )

    warp_scan = capability_for("scan", "warp")
    assert warp_scan.planned_target is GroupLoweringTarget.CUB_WARP
    assert warp_scan.group_first_readiness is GroupFirstReadiness.READY
    assert warp_scan.planned_api is not None
    assert warp_scan.planned_api.entity == "::cub::WarpScan"
    assert warp_scan.group_first_provenance is warp_scan.builtin_scoped_provenance
    assert warp_scan.group_first_provenance.kind is ProvenanceKind.CUB_PUBLIC
    assert [
        selector.name for selector in warp_scan.builtin_scoped_selector_support
    ] == ["threads_in_warp", "valid_items"]
    assert len(warp_scan.alternate_builtin_scoped_routes) == 2
    assert all(
        route.target is GroupLoweringTarget.CUB_WARP
        and route.api.entity == "::cub::WarpScan"
        for route in warp_scan.alternate_builtin_scoped_routes
    )
    assert [
        route.provenance.kind for route in warp_scan.alternate_builtin_scoped_routes
    ] == [
        ProvenanceKind.CUB_PUBLIC,
        ProvenanceKind.CUB_PUBLIC_WITH_GENERATED_ADAPTER,
    ]

    warp_exchange = capability_for("exchange", "warp")
    assert warp_exchange.planned_target is GroupLoweringTarget.CUB_WARP
    assert warp_exchange.group_first_readiness is GroupFirstReadiness.READY
    assert warp_exchange.planned_api is not None
    assert warp_exchange.planned_api.entity == "::cub::WarpExchange"
    assert warp_exchange.builtin_scoped_provenance.kind is ProvenanceKind.CUB_PUBLIC
    assert warp_exchange.builtin_scoped_provenance.api == warp_exchange.planned_api
    assert warp_exchange.group_first_provenance.kind is ProvenanceKind.CUB_PUBLIC
    assert "logical/scatter WarpExchange compatibility routes retain x4" in (
        warp_exchange.builtin_scoped_operand_forms[0].note
    )
    assert warp_exchange.planned_api.methods == (
        "StripedToBlocked",
        "BlockedToStriped",
    )
    assert warp_scan.planned_api.methods == (
        "ExclusiveSum",
        "ExclusiveScan",
        "InclusiveSum",
        "InclusiveScan",
    )
    assert group_first_planner_models_binding("warp", "exchange_blocked_to_striped")
    assert not group_first_planner_models_binding("warp", "exchange_scatter_to_striped")
    assert [
        selector.name for selector in warp_exchange.builtin_scoped_selector_support
    ] == ["threads_in_warp", "mode"]
    logical_exchange, scatter_exchange = warp_exchange.alternate_builtin_scoped_routes
    assert logical_exchange.name == "logical_warp_x4_compatibility"
    assert scatter_exchange.name == "scatter_x4_compatibility"
    assert logical_exchange.cases[0].operand_forms[0].max_items_per_thread == 4
    assert scatter_exchange.cases[0].operand_forms[0].max_items_per_thread == 4
    assert not logical_exchange.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "logical", "mode": "striped_to_blocked"},
    )
    assert logical_exchange.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "logical", "mode": "striped_to_blocked"},
        items_per_thread=4,
    )
    assert not logical_exchange.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "logical", "mode": "striped_to_blocked"},
        items_per_thread=5,
    )
    assert scatter_exchange.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "physical", "mode": "scatter_to_striped"},
        items_per_thread=4,
    )
    assert not scatter_exchange.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "physical", "mode": "scatter_to_striped"},
        items_per_thread=5,
    )

    warp_merge_sort = capability_for("merge_sort", "warp")
    assert warp_merge_sort.group_first_readiness is GroupFirstReadiness.READY
    assert warp_merge_sort.unsupported_reason is None
    assert warp_merge_sort.remaining_group_first_stages == ()
    assert tuple(form.kind for form in warp_merge_sort.group_first_operand_forms) == (
        OperandKind.THREAD_DATA,
    )
    assert warp_merge_sort.group_first_provenance is not None
    assert warp_merge_sort.group_first_provenance.kind is ProvenanceKind.CUB_PUBLIC
    assert [
        selector.name for selector in warp_merge_sort.builtin_scoped_selector_support
    ] == ["threads_in_warp"]
    logical_merge_sort = warp_merge_sort.alternate_builtin_scoped_routes[0]
    assert logical_merge_sort.cases[0].name == "logical_full_or_partial"
    assert logical_merge_sort.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "logical"},
    )
    assert (
        logical_merge_sort.provenance.kind
        is ProvenanceKind.CUB_PUBLIC_WITH_GENERATED_ADAPTER
    )


@pytest.mark.parametrize(
    ("group_kind", "group", "cub_target"),
    [
        (GroupKind.BLOCK, this_block(), GroupLoweringTarget.CUB_BLOCK),
        (GroupKind.WARP, this_warp(), GroupLoweringTarget.CUB_WARP),
    ],
)
def test_reduce_alternate_route_cases_match_every_public_selector_combination(
    group_kind,
    group,
    cub_target,
):
    capability = capability_for(OperationFamily.REDUCE, group_kind)
    routes = capability.alternate_group_first_routes
    algorithms = {
        "omitted": None,
        "raking_commutative_only": (BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY),
        "raking": BlockReduceAlgorithm.RAKING,
        "warp_reductions": BlockReduceAlgorithm.WARP_REDUCTIONS,
    }
    valid_bindings = {
        "omitted": ArgumentBinding.omitted(),
        "static": ArgumentBinding.static(17),
        "runtime": ArgumentBinding.runtime(),
    }
    selected_algorithms = (
        algorithms if group_kind is GroupKind.BLOCK else {"omitted": None}
    )

    for operand_kind, broadcast, valid_mode, algorithm_name in itertools.product(
        (OperandKind.SCALAR, OperandKind.THREAD_DATA),
        (True, False),
        valid_bindings,
        selected_algorithms,
    ):
        all_selectors = {
            "broadcast": str(broadcast).lower(),
            "valid_items": valid_mode,
            "algorithm": algorithm_name,
        }
        selectors = {
            selector.name: all_selectors[selector.name]
            for selector in capability.group_first_selector_support
        }
        matching_routes = [
            route
            for route in routes
            if route.matches(
                operand_kind,
                selectors,
                operator_commutative=True,
            )
        ]
        value_kind = "scalar" if operand_kind is OperandKind.SCALAR else "array"
        try:
            primitive = make_reduce_semantics(
                dtype="int",
                operation="sum",
                value_kind=value_kind,
                items_per_thread=1 if value_kind == "scalar" else 2,
                valid_items=valid_bindings[valid_mode],
            )
        except ValueError as exc:
            assert "valid_items is not supported for array inputs" in str(exc)
            assert matching_routes == []
            continue

        plan = plan_group_primitive(
            make_group_primitive_call(
                group,
                GroupReduceSemantics(
                    primitive,
                    broadcast=broadcast,
                    cub_algorithm=algorithms[algorithm_name],
                ),
            ),
            LaunchFacts(exact_block_dim=64),
        )
        if plan.target is cub_target:
            assert [route.target for route in matching_routes] == [cub_target]
        else:
            assert matching_routes == []


def test_commutative_only_reduce_route_requires_operator_proof():
    capability = capability_for(OperationFamily.REDUCE, GroupKind.BLOCK)
    route = capability.alternate_group_first_routes[0]
    selectors = {
        "broadcast": "false",
        "valid_items": "omitted",
        "algorithm": "raking_commutative_only",
    }

    assert not route.matches(OperandKind.SCALAR, selectors)
    assert not route.matches(
        OperandKind.SCALAR,
        selectors,
        operator_commutative=False,
    )
    assert route.matches(
        OperandKind.SCALAR,
        selectors,
        operator_commutative=True,
    )

    primitive = make_reduce_semantics(
        dtype="int",
        operation="reduce",
        value_kind="scalar",
        items_per_thread=1,
        reduce_operator=CxxOperator("custom_noncommutative", "int"),
    )
    plan = plan_group_primitive(
        make_group_primitive_call(
            this_block(),
            GroupReduceSemantics(
                primitive,
                broadcast=False,
                cub_algorithm=BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
            ),
        ),
        LaunchFacts(exact_block_dim=64),
    )
    assert plan.target is GroupLoweringTarget.UNSUPPORTED


def test_scoped_reduce_routes_record_cudax_defaults_and_cub_exceptions():
    block = capability_for(OperationFamily.REDUCE, GroupKind.BLOCK)
    block_route = block.alternate_builtin_scoped_routes[0]
    assert not block_route.matches(
        OperandKind.SCALAR,
        {"valid_items": "omitted", "algorithm": "omitted"},
        operator_commutative=True,
    )
    assert block_route.matches(
        OperandKind.SCALAR,
        {"valid_items": "runtime", "algorithm": "raking"},
        operator_commutative=True,
    )
    assert block_route.matches(
        OperandKind.THREAD_DATA,
        {"valid_items": "omitted", "algorithm": "warp_reductions"},
        operator_commutative=True,
    )
    assert not block_route.matches(
        OperandKind.THREAD_DATA,
        {"valid_items": "static", "algorithm": "omitted"},
        operator_commutative=True,
    )
    commutative_selectors = {
        "valid_items": "omitted",
        "algorithm": "raking_commutative_only",
    }
    assert not block_route.matches(OperandKind.SCALAR, commutative_selectors)
    assert block_route.matches(
        OperandKind.SCALAR,
        commutative_selectors,
        operator_commutative=True,
    )

    warp = capability_for(OperationFamily.REDUCE, GroupKind.WARP)
    physical_route, logical_route = warp.alternate_builtin_scoped_routes
    assert not physical_route.matches(
        OperandKind.SCALAR,
        {"threads_in_warp": "physical", "valid_items": "omitted"},
    )
    assert physical_route.matches(
        OperandKind.SCALAR,
        {"threads_in_warp": "physical", "valid_items": "runtime"},
    )
    assert not physical_route.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "physical", "valid_items": "static"},
    )
    assert logical_route.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "logical", "valid_items": "omitted"},
    )
    assert logical_route.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "logical", "valid_items": "static"},
    )


def test_builtin_scoped_provenance_does_not_confuse_models_with_implementation():
    radix_rank = capability_for(OperationFamily.RADIX_RANK, "block")
    radix_sort = capability_for(OperationFamily.RADIX_SORT, "block")
    for capability in (radix_rank, radix_sort):
        assert capability.builtin_scoped_provenance.kind is ProvenanceKind.CUB_PUBLIC
        assert capability.group_first_provenance is not None
        assert capability.group_first_provenance.kind is ProvenanceKind.CUB_PUBLIC
    assert radix_rank.group_first_readiness is GroupFirstReadiness.READY
    assert radix_rank.unsupported_reason is None
    assert radix_rank.remaining_group_first_stages == ()
    assert radix_sort.group_first_readiness is GroupFirstReadiness.READY
    assert radix_sort.unsupported_reason is None
    assert radix_sort.remaining_group_first_stages == ()
    assert radix_rank.planned_api is not None
    assert radix_rank.planned_api.entity == "::cub::BlockRadixRank"
    assert radix_rank.planned_api.methods == ("RankKeys",)
    assert radix_sort.planned_api is not None
    assert radix_sort.planned_api.entity == "::cub::BlockRadixSort"
    assert radix_sort.planned_api.methods == ("Sort", "SortDescending")
    adjacent_difference = capability_for(
        OperationFamily.ADJACENT_DIFFERENCE,
        "block",
    )
    assert (
        adjacent_difference.builtin_scoped_provenance.kind is ProvenanceKind.CUB_PUBLIC
    )
    assert adjacent_difference.group_first_provenance is not None
    assert adjacent_difference.group_first_provenance.kind is ProvenanceKind.CUB_PUBLIC
    assert adjacent_difference.planned_api is not None
    assert adjacent_difference.planned_api.methods == (
        "SubtractLeft",
        "SubtractLeftPartialTile",
        "SubtractRight",
        "SubtractRightPartialTile",
    )
    expected_adjacent_selectors = [
        ("direction", ("left", "right")),
        ("valid_items", ("omitted", "runtime")),
        ("tile_predecessor_item", ("omitted", "runtime")),
        ("tile_successor_item", ("omitted", "runtime")),
    ]
    assert [
        (selector.name, selector.accepted_values)
        for selector in adjacent_difference.group_first_selector_support
    ] == expected_adjacent_selectors
    assert [
        (selector.name, selector.accepted_values)
        for selector in adjacent_difference.builtin_scoped_selector_support
    ] == expected_adjacent_selectors
    for family in (
        OperationFamily.DISCONTINUITY,
        OperationFamily.HISTOGRAM,
        OperationFamily.RUN_LENGTH_DECODE,
        OperationFamily.SHUFFLE,
    ):
        converted = capability_for(family, "block")
        assert converted.builtin_scoped_provenance.kind is ProvenanceKind.CUB_PUBLIC
        assert converted.group_first_provenance is not None
        assert converted.group_first_provenance.kind is ProvenanceKind.CUB_PUBLIC
    for family in (
        OperationFamily.DISCONTINUITY,
        OperationFamily.HISTOGRAM,
        OperationFamily.RUN_LENGTH_DECODE,
        OperationFamily.SHUFFLE,
    ):
        assert (
            capability_for(family, "block").group_first_readiness
            is GroupFirstReadiness.READY
        )
    discontinuity = capability_for(OperationFamily.DISCONTINUITY, "block")
    expected_discontinuity_selectors = [
        ("mode", ("heads", "tails", "heads_and_tails")),
        ("tile_predecessor_item", ("omitted", "runtime")),
        ("tile_successor_item", ("omitted", "runtime")),
    ]
    assert [
        (selector.name, selector.accepted_values)
        for selector in discontinuity.group_first_selector_support
    ] == expected_discontinuity_selectors
    assert [
        (selector.name, selector.accepted_values)
        for selector in discontinuity.builtin_scoped_selector_support
    ] == expected_discontinuity_selectors

    shuffle = capability_for(OperationFamily.SHUFFLE, "block")
    expected_shuffle_selectors = [
        ("mode", ("offset", "rotate", "up", "down")),
        ("distance", ("runtime", "unit")),
        ("block_prefix", ("omitted", "output")),
        ("block_suffix", ("omitted", "output")),
    ]
    assert [
        (selector.name, selector.accepted_values)
        for selector in shuffle.group_first_selector_support
    ] == expected_shuffle_selectors
    assert [
        (selector.name, selector.accepted_values)
        for selector in shuffle.builtin_scoped_selector_support
    ] == expected_shuffle_selectors
    assert (
        capability_for(OperationFamily.EXCHANGE, "block").builtin_scoped_provenance.kind
        is ProvenanceKind.CUB_PUBLIC
    )
    block_merge_sort = capability_for(OperationFamily.MERGE_SORT, "block")
    assert block_merge_sort.builtin_scoped_provenance.kind is ProvenanceKind.CUB_PUBLIC
    assert block_merge_sort.group_first_provenance is not None
    assert block_merge_sort.group_first_provenance.kind is ProvenanceKind.CUB_PUBLIC

    for group in ("block", "warp"):
        scan = capability_for("scan", group)
        assert scan.group_first_provenance is scan.builtin_scoped_provenance
        assert scan.group_first_provenance.kind is ProvenanceKind.CUB_PUBLIC
        assert scan.group_first_provenance.api is scan.planned_api

    for group in ("block", "warp"):
        reduce = capability_for("reduce", group)
        assert reduce.builtin_scoped_provenance.kind is ProvenanceKind.CUDAX_PUBLIC
        assert reduce.alternate_builtin_scoped_routes
        assert all(
            route.api.entity in {"::cub::BlockReduce", "::cub::WarpReduce"}
            for route in reduce.alternate_builtin_scoped_routes
        )

    row_reduce = capability_for("row_reduce", "block")
    assert row_reduce.builtin_scoped_provenance.kind is ProvenanceKind.CUB_NOT_IN_TREE
    assert row_reduce.builtin_scoped_provenance.api is not None
    assert row_reduce.builtin_scoped_provenance.api.headers == (
        "cub/block/block_row_reduce.cuh",
    )
    assert (
        row_reduce.builtin_scoped_provenance.api.entity
        == "::cub::BlockRowReduceWarpBroadcast"
    )
    assert row_reduce.builtin_scoped_provenance.api.methods == ("Sum",)
    assert row_reduce.builtin_scoped_provenance.api.stability is ApiStability.UNVERIFIED
    assert (
        row_reduce.builtin_scoped_provenance.api.availability
        is ApiAvailability.NOT_IN_TREE
    )
    assert (
        row_reduce.group_first_readiness
        is GroupFirstReadiness.BLOCKED_PLANNER_AND_DEPENDENCY
    )

    topk = capability_for("topk", "block")
    assert topk.builtin_scoped_provenance.kind is ProvenanceKind.CUB_DETAIL
    assert topk.builtin_scoped_provenance.api is not None
    assert topk.builtin_scoped_provenance.api.stability is ApiStability.DETAIL
    assert topk.planned_target is GroupLoweringTarget.CUB_BLOCK
    assert topk.group_first_readiness is GroupFirstReadiness.READY
    assert topk.remaining_group_first_stages == ()
    assert topk.group_first_provenance.kind is ProvenanceKind.CUB_DETAIL
    assert topk.group_first_provenance.api is topk.planned_api

    for group in ("block", "warp"):
        for family in ("load", "store"):
            capability = capability_for(family, group)
            assert (
                capability.builtin_scoped_provenance.kind is ProvenanceKind.CUB_PUBLIC
            )
            assert capability.builtin_scoped_provenance.api is capability.planned_api
            assert [
                adapter.name for adapter in capability.scoped_payload_adapter_provenance
            ] == ["cute_indexing", "prims_array"]
            assert all(
                adapter.provenance.kind is ProvenanceKind.PAYLOAD_ADAPTER
                and adapter.provenance.api is None
                for adapter in capability.scoped_payload_adapter_provenance
            )
            assert capability.group_first_readiness is GroupFirstReadiness.READY
            assert capability.remaining_group_first_stages == ()
            assert capability.group_first_provenance.kind is ProvenanceKind.CUB_PUBLIC
            assert capability.group_first_provenance.api is capability.planned_api
            expected_target = (
                GroupLoweringTarget.CUB_BLOCK
                if group == "block"
                else GroupLoweringTarget.CUB_WARP
            )
            assert capability.planned_target is expected_target
            expected_method = "Load" if family == "load" else "Store"
            assert capability.planned_api.methods == (expected_method,)


def test_warp_operand_forms_record_actual_scan_and_exchange_limits():
    scan = capability_for("scan", "warp")
    scan_thread_data = next(
        form
        for form in scan.builtin_scoped_operand_forms
        if form.kind is OperandKind.THREAD_DATA
    )
    assert scan_thread_data.min_items_per_thread == 1
    assert scan_thread_data.max_items_per_thread is None
    assert scan.group_first_operand_forms[0].kind is OperandKind.SCALAR
    assert len(scan.group_first_operand_forms) == 1
    scalar_compatibility, thread_data_compatibility = (
        scan.alternate_builtin_scoped_routes
    )
    canonical_selectors = {
        "threads_in_warp": "physical",
        "valid_items": "omitted",
    }
    assert not scalar_compatibility.matches(OperandKind.SCALAR, canonical_selectors)
    assert not thread_data_compatibility.matches(
        OperandKind.SCALAR, canonical_selectors
    )
    assert thread_data_compatibility.matches(
        OperandKind.THREAD_DATA, canonical_selectors
    )
    assert scalar_compatibility.matches(
        OperandKind.SCALAR,
        {"threads_in_warp": "physical", "valid_items": "static"},
    )
    assert scalar_compatibility.matches(
        OperandKind.SCALAR,
        {"threads_in_warp": "logical", "valid_items": "omitted"},
    )
    assert thread_data_compatibility.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "logical", "valid_items": "runtime"},
    )

    exchange = capability_for("exchange", "warp")
    assert len(exchange.builtin_scoped_operand_forms) == 1
    exchange_thread_data = exchange.builtin_scoped_operand_forms[0]
    assert exchange_thread_data.kind is OperandKind.THREAD_DATA
    assert exchange_thread_data.min_items_per_thread == 1
    assert (
        exchange_thread_data.max_items_per_thread == MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD
    )
    assert (
        exchange.group_first_operand_forms[0].max_items_per_thread
        == MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD
    )
    logical_exchange, scatter_exchange = exchange.alternate_builtin_scoped_routes
    assert logical_exchange.name == "logical_warp_x4_compatibility"
    assert scatter_exchange.name == "scatter_x4_compatibility"
    assert logical_exchange.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "logical", "mode": "blocked_to_striped"},
        items_per_thread=4,
    )
    assert not logical_exchange.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "logical", "mode": "blocked_to_striped"},
        items_per_thread=5,
    )
    assert scatter_exchange.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "logical", "mode": "scatter_to_striped"},
        items_per_thread=4,
    )
    assert not scatter_exchange.matches(
        OperandKind.THREAD_DATA,
        {"threads_in_warp": "logical", "mode": "scatter_to_striped"},
        items_per_thread=5,
    )


def test_warp_operand_forms_match_planner_and_builtin_provider_source():
    launch = LaunchFacts(exact_block_dim=64)
    scalar_scan = plan_group_primitive(
        make_group_primitive_call(
            this_warp(),
            GroupScanSemantics(
                make_scan_semantics(
                    dtype="int",
                    mode="inclusive",
                    value_kind="scalar",
                    items_per_thread=1,
                )
            ),
        ),
        launch,
    )
    array_scan = plan_group_primitive(
        make_group_primitive_call(
            this_warp(),
            GroupScanSemantics(
                make_scan_semantics(
                    dtype="int",
                    mode="inclusive",
                    value_kind="array",
                    items_per_thread=2,
                )
            ),
        ),
        launch,
    )
    five_item_exchange = plan_group_primitive(
        make_group_primitive_call(
            this_warp(),
            GroupExchangeSemantics(
                make_block_exchange_semantics(
                    dtype="int",
                    mode="blocked_to_striped",
                    items_per_thread=5,
                )
            ),
        ),
        launch,
    )
    scatter_exchange = plan_group_primitive(
        make_group_primitive_call(
            this_warp(),
            GroupExchangeSemantics(
                make_block_exchange_semantics(
                    dtype="int",
                    mode="scatter_to_striped",
                    items_per_thread=1,
                    rank_dtype="int",
                )
            ),
        ),
        launch,
    )

    assert scalar_scan.target is GroupLoweringTarget.CUB_WARP
    assert array_scan.target is GroupLoweringTarget.UNSUPPORTED
    assert five_item_exchange.target is GroupLoweringTarget.CUB_WARP
    assert scatter_exchange.target is GroupLoweringTarget.CUB_WARP

    compatibility_provider_path = (
        SOURCE_ROOT / "cuda" / "coop" / "cutlass" / "_dsl" / "warp" / "_provider.py"
    )
    group_provider_path = (
        SOURCE_ROOT / "cuda" / "coop" / "cutlass" / "_dsl" / "_cub_exchange_provider.py"
    )
    scan_source = _function_source(
        compatibility_provider_path,
        "_render_cub_warp_scan",
    )
    compatibility_exchange_source = _function_source(
        compatibility_provider_path,
        "provider_exchange",
    )
    group_exchange_source = _function_source(group_provider_path, "provider_exchange")
    assert "if request.items_per_thread > 1:" in scan_source
    assert "thread_total" in scan_source
    assert "result_items" in scan_source
    assert "cub::WarpScan" in scan_source
    assert "if value.items_per_thread > 4:" in compatibility_exchange_source
    assert "if value.items_per_thread > 4:" not in group_exchange_source


def test_every_modeled_operand_form_selects_the_recorded_planner_target():
    launch = LaunchFacts(exact_block_dim=64)
    modeled = [
        capability
        for capability in CAPABILITIES
        if capability.family
        in {OperationFamily.REDUCE, OperationFamily.SCAN, OperationFamily.EXCHANGE}
        and capability.group_first_operand_forms
    ]

    for capability in modeled:
        group = this_block() if capability.group is GroupKind.BLOCK else this_warp()
        for form in capability.group_first_operand_forms:
            items_per_thread = 1 if form.kind is OperandKind.SCALAR else 2
            value_kind = "scalar" if form.kind is OperandKind.SCALAR else "array"
            if capability.family is OperationFamily.REDUCE:
                operation = GroupReduceSemantics(
                    make_reduce_semantics(
                        dtype="int",
                        operation="sum",
                        value_kind=value_kind,
                        items_per_thread=items_per_thread,
                    )
                )
            elif capability.family is OperationFamily.SCAN:
                operation = GroupScanSemantics(
                    make_scan_semantics(
                        dtype="int",
                        mode="inclusive",
                        value_kind=value_kind,
                        items_per_thread=items_per_thread,
                    )
                )
            else:
                operation = GroupExchangeSemantics(
                    make_block_exchange_semantics(
                        dtype="int",
                        mode="blocked_to_striped",
                        items_per_thread=items_per_thread,
                    )
                )

            plan = plan_group_primitive(
                make_group_primitive_call(group, operation),
                launch,
            )

            assert plan.target is capability.planned_target
            if plan.target is not GroupLoweringTarget.CUDAX_GROUP:
                assert plan.implementation.method_name in capability.planned_api.methods


def test_in_tree_cpp_api_records_name_checked_in_headers():
    repository_root = SOURCE_ROOT.parents[1]
    apis = set()
    for capability in CAPABILITIES:
        candidates = [
            capability.planned_api,
            (
                None
                if capability.builtin_scoped_provenance is None
                else capability.builtin_scoped_provenance.api
            ),
            (
                None
                if capability.group_first_provenance is None
                else capability.group_first_provenance.api
            ),
            *(route.api for route in capability.alternate_group_first_routes),
            *(route.api for route in capability.alternate_builtin_scoped_routes),
        ]
        apis.update(
            api
            for api in candidates
            if api is not None and api.availability is ApiAvailability.IN_TREE
        )

    for api in apis:
        for header in api.headers:
            header_path = (
                repository_root / "cub" / header
                if header.startswith("cub/")
                else repository_root / "cudax" / "include" / header
            )
            assert header_path.is_file(), header

    row_reduce = capability_for("row_reduce", "block")
    row_api = row_reduce.builtin_scoped_provenance.api
    assert row_api is not None
    assert row_api.availability is ApiAvailability.NOT_IN_TREE
    assert not (repository_root / "cub" / row_api.headers[0]).exists()


def test_capability_registry_imports_with_only_the_validated_runtime_boundary():
    script = textwrap.dedent(
        """
        import sys
        import types

        cutlass = types.ModuleType("cutlass")
        cutlass.__path__ = []
        cutlass_dsl = types.ModuleType("cutlass.cutlass_dsl")

        class CuTeDSL:
            _instance = None

            @classmethod
            def _get_dsl(cls):
                if cls._instance is None:
                    cls._instance = cls()
                return cls._instance

            def register_trace_context_factory(self, factory):
                self.trace_context_factory = factory

            def register_trace_finalize_hook(self, hook):
                self.trace_finalize_hook = hook

        cutlass_dsl.CuTeDSL = CuTeDSL
        cute = types.ModuleType("cutlass.cute")
        cute._get_launch_facts = lambda: {}
        base_dsl = types.ModuleType("cutlass.base_dsl")
        base_dsl.__path__ = []
        compiler = types.ModuleType("cutlass.base_dsl.compiler")
        compiler.LinkLibraries = type(
            "LinkLibraries",
            (),
            {"_option_name": "link-libraries"},
        )
        compiler.GPUArch = type("GPUArch", (), {})
        cutlass.cutlass_dsl = cutlass_dsl
        cutlass.cute = cute
        cutlass.base_dsl = base_dsl
        base_dsl.compiler = compiler
        sys.modules["cutlass"] = cutlass
        sys.modules["cutlass.cutlass_dsl"] = cutlass_dsl
        sys.modules["cutlass.cute"] = cute
        sys.modules["cutlass.base_dsl"] = base_dsl
        sys.modules["cutlass.base_dsl.compiler"] = compiler

        from cuda.coop._core.group_dispatch import GroupLoweringTarget
        import cuda.coop.cutlass._capabilities as capabilities

        reduce = capabilities.capability_for("reduce", "block")
        assert reduce.planned_target is GroupLoweringTarget.CUDAX_GROUP
        assert "cutlass.cutlass_dsl" in sys.modules
        assert "cutlass.cute" in sys.modules
        assert "cutlass.base_dsl.compiler" in sys.modules
        assert "torch" not in sys.modules
        assert "numpy" not in sys.modules
        assert "numba" not in sys.modules
        """
    )
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SOURCE_ROOT}{os.pathsep}{python_path}" if python_path else str(SOURCE_ROOT)
    )
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
