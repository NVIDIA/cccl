# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from importlib import import_module

import numpy as np
import pytest

from cuda.coop._core import (
    ArgumentBinding,
    ArgumentKind,
    BlockScanAlgorithm,
    CxxFunction,
    CxxOperator,
    Dependency,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupScanSemantics,
    LaunchFacts,
    ParameterRole,
    PreconditionEnforcement,
    PythonOperator,
    ResultOwnership,
    ResultVisibility,
    StatefulOperator,
    StorageOwnership,
    SynchronizationScope,
    UnsupportedReasonCode,
    make_group_primitive_call,
    make_scan_semantics,
    plan_group_primitive,
    this_block,
    this_cluster,
    this_thread,
    this_warp,
)
from cuda.coop._core.thread_group import CoopCompilerContextRequiredError
from tests.support.group_planning import _plan, _scan


class _ThreadData:
    def __init__(self, items_per_thread=2, *, dtype=np.float32, length=None):
        self.items_per_thread = items_per_thread
        self.dtype = dtype
        self._items = [np.float32(0)] * (items_per_thread if length is None else length)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class _TempStorage:
    size_in_bytes = 128
    alignment = 16
    auto_sync = True
    sharing = "shared"


@pytest.mark.parametrize(
    ("group", "operand_kind", "items", "target", "scope", "instances"),
    [
        (
            this_block(),
            GroupOperandKind.SCALAR,
            1,
            GroupLoweringTarget.CUB_BLOCK,
            SynchronizationScope.BLOCK,
            1,
        ),
        (
            this_block(),
            GroupOperandKind.ARRAY,
            4,
            GroupLoweringTarget.CUB_BLOCK,
            SynchronizationScope.BLOCK,
            1,
        ),
        (
            this_warp(),
            GroupOperandKind.SCALAR,
            1,
            GroupLoweringTarget.CUB_WARP,
            SynchronizationScope.WARP,
            4,
        ),
        (
            this_warp().group_by(8),
            GroupOperandKind.SCALAR,
            1,
            GroupLoweringTarget.CUB_WARP,
            SynchronizationScope.WARP,
            16,
        ),
    ],
)
def test_scan_plans_out_of_place_results_and_scope_storage(
    group,
    operand_kind,
    items,
    target,
    scope,
    instances,
):
    operation = _scan(
        mode="inclusive",
        operand_kind=operand_kind,
        items_per_thread=items,
    )
    plan = _plan(group, operation, 128)

    assert operation.result_visibility is ResultVisibility.PER_MEMBER
    assert operation.returns_value
    assert plan.target is target
    assert plan.result.primary.name == "value"
    assert plan.result.primary.ownership is ResultOwnership.EACH_MEMBER
    assert plan.result.primary.operand_kind is operand_kind
    assert plan.result.primary.items_per_member == items
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.address_space == "shared"
    assert plan.temp_storage.instances == instances
    assert plan.synchronization.storage_reuse_barrier is scope
    assert plan.topology.execution_scope is scope


def test_aggregate_is_a_scalar_all_member_side_output_excluding_initial():
    operation = _scan(
        dtype="float32",
        mode="exclusive",
        initial_value=CxxFunction(
            "1.0F",
            Dependency("T"),
            name="initial_value",
        ),
        aggregate=True,
    )
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(64))

    assert [value.name for value in plan.result.values] == ["value", "aggregate"]
    aggregate = plan.result.values[1]
    assert aggregate.dtype == "float32"
    assert aggregate.visibility is ResultVisibility.ALL_MEMBERS
    assert aggregate.ownership is ResultOwnership.EACH_MEMBER
    assert aggregate.operand_kind is GroupOperandKind.SCALAR
    assert aggregate.items_per_member == 1
    assert plan.implementation.metadata["aggregate_excludes_initial"]
    assert [item.name for item in call.argument_classifications] == [
        "value",
        "initial_value",
        "aggregate_output",
        "mode",
        "algorithm",
    ]
    aggregate_parameter = plan.implementation.parameters[0][-1]
    assert aggregate_parameter.name == "block_aggregate"
    assert aggregate_parameter.role is ParameterRole.OUTPUT
    assert aggregate_parameter.is_return is False


def test_partial_logical_warp_scan_bounds_and_declares_runtime_controls():
    operation = _scan(
        mode="inclusive",
        valid_items=ArgumentBinding.runtime(),
        aggregate=True,
    )
    call = make_group_primitive_call(this_warp().group_by(8), operation)
    plan = plan_group_primitive(call, LaunchFacts(64))

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.implementation.method_name == "InclusiveScanPartial"
    assert plan.implementation.template_arguments["VIRTUAL_WARP_THREADS"] == 8
    assert plan.topology.instances == 8
    assert plan.temp_storage.instances == 8
    assert plan.temp_storage.instance_index == "linear_thread_rank / 8"
    assert plan.participation.uniform_arguments == ("valid_items",)
    assert plan.participation.valid_member_selection == (
        "first valid_items lanes by linear group rank"
    )
    precondition = plan.participation.argument_preconditions[0]
    assert (precondition.minimum, precondition.maximum) == (1, 8)
    assert precondition.enforcement is PreconditionEnforcement.CALLER
    assert [item.name for item in call.argument_classifications] == [
        "value",
        "valid_items",
        "aggregate_output",
        "mode",
        "algorithm",
    ]
    valid_items = next(
        item for item in plan.implementation.parameters[0] if item.name == "valid_items"
    )
    assert valid_items.dtype.name == "int32"


def test_partial_exclusive_sum_is_canonicalized_to_typed_zero():
    operation = _scan(
        mode="exclusive",
        valid_items=ArgumentBinding.static(np.int32(5)),
    )
    plan = _plan(this_warp().group_by(8), operation, 64)

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.implementation.method_name == "ExclusiveScanPartial"
    assert plan.call.operation.initial_value == CxxFunction(
        "{T}{0}",
        Dependency("T"),
        name="initial_value",
    )
    assert plan.call.operation.scan_operator == CxxOperator(
        "::cuda::std::plus<T>",
        Dependency("T"),
        name="scan_op",
    )
    assert plan.call.operation.valid_items == ArgumentBinding.static(5)
    assert [item.name for item in plan.call.argument_classifications] == [
        "value",
        "scan_op",
        "initial_value",
        "valid_items",
        "mode",
        "algorithm",
    ]
    precondition = plan.participation.argument_preconditions[0]
    assert precondition.enforcement is PreconditionEnforcement.PLANNER_VALIDATED


def test_default_block_algorithm_is_canonical_in_plan_identity():
    omitted = _plan(this_block(), _scan(mode="inclusive"))
    explicit = _plan(
        this_block(),
        _scan(mode="inclusive", cub_algorithm=BlockScanAlgorithm.RAKING),
    )

    assert omitted.call.operation.cub_algorithm is BlockScanAlgorithm.RAKING
    assert omitted.semantic_key == explicit.semantic_key
    assert omitted.artifact_key == explicit.artifact_key
    assert omitted == explicit


@pytest.mark.parametrize(
    ("block_threads", "supported"),
    [(16, False), (48, False), (32, True), (64, True)],
)
def test_block_warp_scans_requires_a_complete_warp_multiple(
    block_threads,
    supported,
):
    plan = _plan(
        this_block(),
        _scan(mode="inclusive", cub_algorithm=BlockScanAlgorithm.WARP_SCANS),
        block_threads,
    )

    if supported:
        assert plan.target is GroupLoweringTarget.CUB_BLOCK
    else:
        assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
        assert "multiple" in plan.unsupported.message


def test_scan_rejects_group_and_operand_variants_without_exact_cub_support():
    warp_array = _plan(
        this_warp(),
        _scan(mode="inclusive", operand_kind="array", items_per_thread=2),
    )
    warp_algorithm = _plan(
        this_warp(),
        _scan(mode="inclusive", cub_algorithm="raking"),
    )
    block_prefix = _plan(
        this_block(),
        _scan(mode="inclusive", valid_items=ArgumentBinding.runtime()),
    )
    thread = _plan(this_thread(), _scan(mode="inclusive"))
    cluster = _plan(this_cluster(), _scan(mode="inclusive"))

    assert warp_array.unsupported.code is UnsupportedReasonCode.OPERAND_FORM
    assert warp_algorithm.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert block_prefix.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert thread.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert cluster.unsupported.code is UnsupportedReasonCode.GROUP_KIND


def test_custom_exclusive_scan_requires_initial_value():
    plan = _plan(
        this_block(),
        _scan(
            scan_operator=CxxOperator(
                "::cuda::maximum<T>",
                Dependency("T"),
                name="scan_op",
            ),
        ),
    )

    assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert "require an initial value" in plan.unsupported.message
    assert "rank zero" in plan.unsupported.message


def test_block_prefix_callback_defines_custom_exclusive_rank_zero():
    def maximum(left, right):
        return left if left > right else right

    def running_prefix(state, aggregate):
        previous = state[0]
        state[0] = maximum(previous, aggregate)
        return previous

    operation = GroupScanSemantics(
        make_scan_semantics(
            dtype="int32",
            mode="exclusive",
            value_kind="scalar",
            items_per_thread=1,
            scan_operator=PythonOperator(
                Dependency("T"),
                (Dependency("T"), Dependency("T")),
                maximum,
                name="scan_op",
            ),
            prefix_callback=StatefulOperator(
                running_prefix,
                state_dtype="int64",
                ret_dtype=Dependency("T"),
                arg_dtypes=(Dependency("T"),),
                name="prefix_op",
            ),
        )
    )
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(64))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.method_name == "ExclusiveScan"
    assert [item.name for item in call.argument_classifications] == [
        "value",
        "scan_op",
        "prefix_op",
        "mode",
        "algorithm",
    ]
    assert [item.kind for item in call.argument_classifications] == [
        ArgumentKind.RUNTIME,
        ArgumentKind.STATIC,
        ArgumentKind.RUNTIME,
        ArgumentKind.STATIC,
        ArgumentKind.STATIC,
    ]
    assert call.argument_classifications[1].role is ParameterRole.OPERATOR
    assert call.argument_classifications[2].role is ParameterRole.STATE
    assert [item.name for item in plan.implementation.parameters[0]] == [
        "temp_storage",
        "input",
        "output",
        "scan_op",
        "prefix_op",
    ]
    assert plan.participation.uniform_arguments == ()


@pytest.mark.parametrize("group", [this_warp(), this_warp().group_by(8)])
def test_prefix_callbacks_are_block_only(group):
    operation = GroupScanSemantics(
        make_scan_semantics(
            dtype="int32",
            mode="inclusive",
            value_kind="scalar",
            items_per_thread=1,
            prefix_callback=PythonOperator(
                Dependency("T"),
                (Dependency("T"),),
                lambda value: value,
                name="prefix_op",
            ),
        )
    )
    plan = _plan(group, operation)

    assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert "only to physical block groups" in plan.unsupported.message


@pytest.mark.parametrize("valid_items", [0, -1, 9])
def test_static_warp_prefix_is_bounded_to_group_width(valid_items):
    operation = _scan(
        mode="inclusive",
        valid_items=ArgumentBinding.static(valid_items),
    )
    if valid_items < 1:
        with pytest.raises(ValueError, match="between 1"):
            _plan(this_warp().group_by(8), operation)
    else:
        with pytest.raises(ValueError, match="between 1"):
            _plan(this_warp().group_by(8), operation)


def test_portable_scan_validates_payload_and_option_matrix(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.scan")
    delegated = object()
    calls = []

    def marker(*args, **kwargs):
        calls.append((args, kwargs))
        return delegated

    monkeypatch.setattr(api, "_group_primitive_marker", marker)
    with dispatch._compiler_scope("test.backend"):
        assert api.inclusive_sum(this_block(), _ThreadData()) is delegated
        assert api.scan(this_warp(), np.float32(1)) is delegated
        assert (
            api.exclusive_scan(
                this_block(),
                np.float32(1),
                scan_op=" maximum ",
                initial_value=0,
                algorithm="raking-memoize",
                temp_storage=_TempStorage(),
            )
            is delegated
        )
        with pytest.raises(TypeError, match="portable numeric scalar"):
            api.inclusive_sum(this_warp(), _ThreadData())
        with pytest.raises(ValueError, match="require initial_value"):
            api.exclusive_scan(this_block(), np.int32(1), scan_op="max")
        with pytest.raises(ValueError, match="only for blocks"):
            api.inclusive_sum(this_warp(), np.int32(1), algorithm="raking")
        with pytest.raises(ValueError, match="only for blocks"):
            api.inclusive_sum(this_warp(), np.int32(1), temp_storage=object())
        with pytest.raises(TypeError, match="must satisfy TempStorageLike"):
            api.inclusive_sum(this_block(), np.int32(1), temp_storage=object())
        with pytest.raises(TypeError, match="value dtypes"):
            api.inclusive_scan(this_block(), np.float32(1), scan_op="bit_and")

    assert len(calls) == 3
    assert calls[2][1]["scan_op"] == "max"
    assert calls[2][1]["algorithm"] == "raking_memoize"


def test_portable_scan_rejects_inclusive_initial_before_delegation(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.scan")
    calls = []

    monkeypatch.setattr(
        api,
        "_group_primitive_marker",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with dispatch._compiler_scope("test.backend"):
        with pytest.raises(ValueError, match="not supported for inclusive"):
            api.scan(this_block(), np.int32(1), mode="inclusive", initial_value=0)

    assert calls == []


def test_portable_scan_defers_to_compiler_activation_and_exports_root():
    import cuda.coop as coop

    api = import_module("cuda.coop._core.api.scan")
    with pytest.raises(CoopCompilerContextRequiredError):
        api.inclusive_sum(this_block(), object())

    for name in (
        "scan",
        "exclusive_scan",
        "inclusive_scan",
        "exclusive_sum",
        "inclusive_sum",
    ):
        assert getattr(coop, name) is getattr(api, name)
        assert name in coop.__all__


def test_portable_surface_keeps_qualified_only_scan_controls_out():
    api = import_module("cuda.coop._core.api.scan")

    with pytest.raises(TypeError, match="aggregate_output"):
        api.inclusive_sum(this_block(), np.int32(1), aggregate_output=_ThreadData(1))
    with pytest.raises(TypeError, match="valid_items"):
        api.inclusive_sum(this_warp(), np.int32(1), valid_items=17)
