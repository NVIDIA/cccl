# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group scan planner contracts."""

from importlib import import_module

import pytest

from cuda.coop._core.thread_group import CoopCompilerContextRequiredError
from tests.support.group_planning import (
    ArgumentBinding,
    BlockScanAlgorithm,
    CxxFunction,
    CxxOperator,
    Dependency,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupScanMode,
    LaunchFacts,
    PreconditionEnforcement,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    UnsupportedReasonCode,
    _plan,
    _scan,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
    this_warp,
)


def test_portable_scan_rejects_inclusive_initial_value_before_delegation(
    monkeypatch,
):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    scan_api = import_module("cuda.coop._core.api.scan")
    calls = []
    delegated = object()

    def marker(*args, **kwargs):
        calls.append((args, kwargs))
        return delegated

    monkeypatch.setattr(dispatch, "_QUALIFIED_BACKEND_MODULE", None)
    with pytest.raises(CoopCompilerContextRequiredError):
        scan_api.scan(this_block(), 1, mode="inclusive", initial_value=0)

    monkeypatch.setattr(scan_api, "_group_primitive_marker", marker)
    with dispatch._compiler_scope("test.backend"):
        with pytest.raises(ValueError, match="not supported for inclusive scans"):
            scan_api.scan(this_block(), 1, mode=" Inclusive ", initial_value=0)
    assert calls == []

    with dispatch._compiler_scope("test.backend"):
        assert scan_api.scan(this_block(), 1, mode="inclusive") is delegated
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("group", "operand_kind", "items", "target", "struct_name"),
    [
        (
            this_block(),
            GroupOperandKind.SCALAR,
            1,
            GroupLoweringTarget.CUB_BLOCK,
            "BlockScan",
        ),
        (
            this_block(),
            GroupOperandKind.ARRAY,
            4,
            GroupLoweringTarget.CUB_BLOCK,
            "BlockScan",
        ),
        (
            this_warp(),
            GroupOperandKind.SCALAR,
            1,
            GroupLoweringTarget.CUB_WARP,
            "WarpScan",
        ),
    ],
)
def test_scan_selects_exact_block_or_scalar_warp_cub(
    group,
    operand_kind,
    items,
    target,
    struct_name,
):
    operation = _scan(
        dtype="int",
        mode=GroupScanMode.EXCLUSIVE,
        operand_kind=operand_kind,
        items_per_thread=items,
        aggregate=True,
    )
    plan = _plan(group, operation, 128)

    assert plan.target is target
    assert plan.implementation.struct_name == struct_name
    assert plan.result.visibility is ResultVisibility.PER_MEMBER
    assert plan.result.operand_kind is operand_kind
    assert plan.result.result_items_per_thread == items
    assert plan.result.has_aggregate
    assert [value.name for value in plan.result.values] == ["value", "aggregate"]
    value, aggregate = plan.result.values
    assert value.dtype == "int"
    assert value.ownership is ResultOwnership.EACH_MEMBER
    assert value.items_per_member == items
    assert aggregate.dtype == "int"
    assert aggregate.visibility is ResultVisibility.ALL_MEMBERS
    assert aggregate.ownership is ResultOwnership.EACH_MEMBER
    assert aggregate.operand_kind is GroupOperandKind.SCALAR
    assert aggregate.items_per_member == 1
    assert aggregate.root_rank is None
    assert plan.provenance.library == "CUB"
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.cpp_type is None
    assert plan.temp_storage.instances is None
    assert plan.temp_storage.instance_index is None


def test_logical_warp_partial_scan_uses_mapped_width_and_aggregate_contract():
    operation = _scan(
        mode="inclusive",
        aggregate=True,
        valid_items=ArgumentBinding.static(5),
    )
    call = make_group_primitive_call(this_warp().group_by(8), operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=64))

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.implementation.struct_name == "WarpScan"
    assert plan.implementation.method_name == "InclusiveScanPartial"
    assert plan.implementation.template_arguments["VIRTUAL_WARP_THREADS"] == 8
    assert plan.result.has_aggregate
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.instances is None
    assert plan.participation.uniform_arguments == ("valid_items",)
    precondition = plan.participation.argument_preconditions[0]
    assert (precondition.minimum, precondition.maximum) == (1, 8)
    assert precondition.enforcement is PreconditionEnforcement.PLANNER_VALIDATED
    assert [item.name for item in call.argument_classifications] == [
        "value",
        "mode",
        "valid_items",
    ]


def test_block_scan_default_algorithm_canonicalizes_in_lowered_artifact():
    default = _plan(this_block(), _scan())
    explicit = _plan(
        this_block(),
        _scan(cub_algorithm=BlockScanAlgorithm.RAKING),
    )

    assert default.call.semantic_key != explicit.call.semantic_key
    assert default.semantic_key == explicit.semantic_key
    assert default.artifact_key == explicit.artifact_key
    assert default == explicit


def test_warp_scan_rejects_multi_item_and_block_only_algorithm_variants():
    multi_item = _plan(
        this_warp(),
        _scan(
            dtype="int",
            mode="inclusive",
            operand_kind="array",
            items_per_thread=2,
        ),
    )
    algorithm = _plan(
        this_warp(),
        _scan(
            dtype="int",
            mode="inclusive",
            operand_kind="scalar",
            items_per_thread=1,
            cub_algorithm="::cub::BLOCK_SCAN_WARP_SCANS",
        ),
    )

    assert multi_item.unsupported.code is UnsupportedReasonCode.OPERAND_FORM
    assert algorithm.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT

    with pytest.raises(ValueError, match="unsupported CUB BlockScan algorithm"):
        _scan(
            dtype="int",
            mode="inclusive",
            operand_kind="scalar",
            items_per_thread=1,
            cub_algorithm="::cub::BLOCK_REDUCE_RAKING",
        )


@pytest.mark.parametrize(
    ("group", "operand_kind", "items_per_thread"),
    [
        (this_block(), "scalar", 1),
        (this_block(), "array", 2),
        (this_warp(), "scalar", 1),
    ],
)
def test_generic_exclusive_scan_without_initial_value_is_unsupported(
    group,
    operand_kind,
    items_per_thread,
):
    plan = _plan(
        group,
        _scan(
            mode="exclusive",
            operand_kind=operand_kind,
            items_per_thread=items_per_thread,
            scan_operator=CxxOperator("::cuda::maximum<>{}", "int"),
        ),
    )

    assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert "require an initial value" in plan.unsupported.message
    assert "rank zero" in plan.unsupported.message


@pytest.mark.parametrize(
    ("block_threads", "is_supported"),
    [(16, False), (48, False), (32, True), (64, True)],
)
def test_block_warp_scans_algorithm_requires_complete_warp_multiple(
    block_threads,
    is_supported,
):
    plan = _plan(
        this_block(),
        _scan(cub_algorithm=BlockScanAlgorithm.WARP_SCANS),
        block_threads,
    )

    if is_supported:
        assert plan.target is GroupLoweringTarget.CUB_BLOCK
        assert plan.unsupported is None
    else:
        assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
        assert "BLOCK_SCAN_WARP_SCANS" in plan.unsupported.message
        assert "multiple" in plan.unsupported.message


@pytest.mark.parametrize(
    ("group", "mode", "operand_kind", "scan_operator"),
    [
        (this_block(), "exclusive", "scalar", None),
        (this_block(), "inclusive", "array", None),
        (
            this_block(),
            "inclusive",
            "scalar",
            CxxOperator("::cuda::maximum<>{}", "int"),
        ),
        (this_warp(), "exclusive", "scalar", None),
    ],
)
def test_scan_rejects_initial_value_without_an_exact_cub_overload(
    group,
    mode,
    operand_kind,
    scan_operator,
):
    operation = _scan(
        dtype="int",
        mode=mode,
        operand_kind=operand_kind,
        items_per_thread=2 if operand_kind == "array" else 1,
        scan_operator=scan_operator,
        initial_value=CxxFunction("0", "int"),
    )
    plan = _plan(group, operation)

    assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT


@pytest.mark.parametrize("group", [this_block(), this_warp()])
def test_custom_exclusive_scan_with_initial_value_is_planned(group):
    operation = _scan(
        dtype="int",
        mode="exclusive",
        operand_kind="scalar",
        items_per_thread=1,
        scan_operator=CxxOperator("::cuda::maximum<>{}", "int"),
        initial_value=CxxFunction("0", "int"),
    )
    plan = _plan(group, operation)

    assert plan.target in {
        GroupLoweringTarget.CUB_BLOCK,
        GroupLoweringTarget.CUB_WARP,
    }
    assert plan.participation.uniform_arguments == ("initial_value",)


def test_scan_prefix_callback_is_a_typed_unsupported_variant():
    from cuda.coop._core import PythonOperator

    operation = _scan(
        dtype="int",
        mode="exclusive",
        operand_kind="scalar",
        items_per_thread=1,
        prefix_callback=PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"),),
            op=lambda value: value,
        ),
    )
    plan = _plan(this_block(), operation)

    assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
