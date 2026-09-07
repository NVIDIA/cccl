# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group scan planner contracts."""

from importlib import import_module

import numpy as np
import pytest

from cuda.coop._core.thread_group import CoopCompilerContextRequiredError
from tests.support.group_planning import (
    ArgumentBinding,
    ArgumentKind,
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


class _ThreadData:
    def __init__(self, items_per_thread=2, *, dtype=np.float32, length=None):
        self.items_per_thread = items_per_thread
        self.dtype = dtype
        self._items = [0] * (items_per_thread if length is None else length)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._items[index] = value


class _ReadonlyThreadData:
    items_per_thread = 2
    dtype = np.float32

    def __init__(self):
        self._items = [np.float32(1), np.float32(2)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class _TempStorage:
    size_in_bytes = 128
    alignment = 16
    auto_sync = True
    sharing = "shared"


def test_portable_scan_validates_payloads_and_option_matrix(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    scan_api = import_module("cuda.coop._core.api.scan")
    calls = []
    delegated = object()

    def marker(*args, **kwargs):
        calls.append((args, kwargs))
        return delegated

    monkeypatch.setattr(scan_api, "_group_primitive_marker", marker)
    with dispatch._compiler_scope("test.backend"):
        assert scan_api.scan(this_warp(), np.float32(1)) is delegated
        with pytest.raises(TypeError, match="value must be a portable numeric scalar"):
            scan_api.scan(this_warp(), _ThreadData())
        assert (
            scan_api.inclusive_sum(
                this_block(),
                _ThreadData(),
                temp_storage=_TempStorage(),
            )
            is delegated
        )
        assert (
            scan_api.exclusive_scan(
                this_block(),
                np.float32(1),
                scan_op="max",
                initial_value=0,
            )
            is delegated
        )
        with pytest.raises(ValueError, match="require initial_value"):
            scan_api.scan(
                this_block(),
                np.float32(1),
                mode="exclusive",
                scan_op="max",
            )
        with pytest.raises(ValueError, match="require initial_value"):
            scan_api.exclusive_scan(
                this_block(),
                np.float32(1),
                scan_op="max",
            )
        with pytest.raises(ValueError, match="only for blocks"):
            scan_api.inclusive_sum(
                this_warp(),
                np.float32(1),
                temp_storage=object(),
            )
        with pytest.raises(ValueError, match="only for blocks"):
            scan_api.inclusive_scan(
                this_warp(),
                np.float32(1),
                algorithm="raking",
            )
        with pytest.raises(TypeError, match="must satisfy TempStorageLike"):
            scan_api.inclusive_sum(
                this_block(),
                np.float32(1),
                temp_storage=object(),
            )
        with pytest.raises(ValueError, match="must match the payload item count"):
            scan_api.inclusive_sum(this_block(), _ThreadData(length=1))
        with pytest.raises(TypeError, match="backend-specific payloads"):
            scan_api.inclusive_sum(this_block(), object())

    assert len(calls) == 3


@pytest.mark.parametrize(
    ("entrypoint", "kwargs"),
    [
        ("scan", {}),
        ("exclusive_sum", {}),
        ("inclusive_sum", {}),
        ("exclusive_scan", {}),
        ("inclusive_scan", {}),
    ],
)
def test_portable_block_scan_accepts_readonly_thread_payloads(
    monkeypatch,
    entrypoint,
    kwargs,
):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    scan_api = import_module("cuda.coop._core.api.scan")
    delegated = object()
    monkeypatch.setattr(
        scan_api,
        "_group_primitive_marker",
        lambda *args, **marker_kwargs: delegated,
    )

    with dispatch._compiler_scope("test.backend"):
        assert (
            getattr(scan_api, entrypoint)(
                this_block(),
                _ReadonlyThreadData(),
                **kwargs,
            )
            is delegated
        )


def test_portable_warp_scan_still_requires_scalar_payload(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    scan_api = import_module("cuda.coop._core.api.scan")
    monkeypatch.setattr(
        scan_api,
        "_group_primitive_marker",
        lambda *args, **kwargs: None,
    )

    with dispatch._compiler_scope("test.backend"):
        with pytest.raises(TypeError, match="value must be a portable numeric scalar"):
            scan_api.inclusive_sum(this_warp(), _ReadonlyThreadData())


@pytest.mark.parametrize("entrypoint", ["scan", "exclusive_scan"])
@pytest.mark.parametrize("initial_value", [object(), _ThreadData()])
def test_portable_scan_rejects_non_scalar_initial_values_before_delegation(
    monkeypatch,
    entrypoint,
    initial_value,
):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    scan_api = import_module("cuda.coop._core.api.scan")
    calls = []
    monkeypatch.setattr(
        scan_api,
        "_group_primitive_marker",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with dispatch._compiler_scope("test.backend"):
        with pytest.raises(TypeError, match="initial_value must be a portable"):
            getattr(scan_api, entrypoint)(
                this_block(),
                np.float32(1),
                initial_value=initial_value,
            )

    assert calls == []


def test_portable_scan_normalizes_and_rejects_operator_tokens(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    scan_api = import_module("cuda.coop._core.api.scan")
    calls = []
    delegated = object()

    def marker(*args, **kwargs):
        calls.append((args, kwargs))
        return delegated

    monkeypatch.setattr(scan_api, "_group_primitive_marker", marker)
    with dispatch._compiler_scope("test.backend"):
        assert (
            scan_api.inclusive_scan(
                this_block(),
                np.int32(1),
                scan_op=" Minimum ",
            )
            is delegated
        )
        with pytest.raises(ValueError, match="scan_op must be one of"):
            scan_api.inclusive_scan(
                this_block(),
                np.int32(1),
                scan_op="unknown",
            )
        with pytest.raises(ValueError, match="algorithm must be one of"):
            scan_api.inclusive_scan(
                this_block(),
                np.int32(1),
                algorithm=["raking"],
            )

    assert len(calls) == 1
    assert calls[0][1]["scan_op"] == "min"


@pytest.mark.parametrize("scan_op", ["bit_and", "bit_or", "bit_xor"])
@pytest.mark.parametrize(
    "value",
    [np.float32(1), _ThreadData(dtype=np.float64)],
)
def test_portable_scan_rejects_bitwise_float_payloads(
    monkeypatch,
    scan_op,
    value,
):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    scan_api = import_module("cuda.coop._core.api.scan")
    monkeypatch.setattr(
        scan_api,
        "_group_primitive_marker",
        lambda *args, **kwargs: None,
    )

    with dispatch._compiler_scope("test.backend"):
        with pytest.raises(TypeError, match="value dtypes"):
            scan_api.inclusive_scan(this_block(), value, scan_op=scan_op)


@pytest.mark.parametrize("scan_op", ["bit_and", "bit_or", "bit_xor"])
def test_portable_scan_accepts_bitwise_integer_payloads(monkeypatch, scan_op):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    scan_api = import_module("cuda.coop._core.api.scan")
    delegated = object()
    monkeypatch.setattr(
        scan_api,
        "_group_primitive_marker",
        lambda *args, **kwargs: delegated,
    )

    with dispatch._compiler_scope("test.backend"):
        assert (
            scan_api.inclusive_scan(this_block(), np.int32(1), scan_op=scan_op)
            is delegated
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
    with pytest.raises(CoopCompilerContextRequiredError):
        scan_api.scan(this_block(), 1, initial_value=object())

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
        valid_items=ArgumentBinding.static(np.int32(5)),
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
    implementation_valid_items = plan.implementation.parameters[0][-2]
    assert implementation_valid_items.name == "valid_items"
    assert implementation_valid_items.argument_kind is ArgumentKind.STATIC
    assert implementation_valid_items.cpp == "5"


def test_group_warp_scan_canonicalizes_static_valid_items_identity():
    group = this_warp().group_by(8)
    plain_operation = _scan(
        mode="inclusive",
        valid_items=ArgumentBinding.static(5),
    )
    numpy_operation = _scan(
        mode="inclusive",
        valid_items=ArgumentBinding.static(np.int32(5)),
    )
    plain_call = make_group_primitive_call(group, plain_operation)
    numpy_call = make_group_primitive_call(group, numpy_operation)
    plain_plan = plan_group_primitive(plain_call, LaunchFacts(exact_block_dim=64))
    numpy_plan = plan_group_primitive(numpy_call, LaunchFacts(exact_block_dim=64))

    assert numpy_operation.valid_items == ArgumentBinding.static(5)
    assert plain_operation.semantic_key == numpy_operation.semantic_key
    assert plain_call.semantic_key == numpy_call.semantic_key
    assert plain_plan.semantic_key == numpy_plan.semantic_key
    assert plain_plan.artifact_key == numpy_plan.artifact_key


def test_block_scan_default_algorithm_canonicalizes_in_lowered_artifact():
    default = _plan(this_block(), _scan())
    explicit = _plan(
        this_block(),
        _scan(cub_algorithm=BlockScanAlgorithm.RAKING),
    )
    portable = _plan(
        this_block(),
        _scan(cub_algorithm="raking"),
    )

    assert default.call.semantic_key != explicit.call.semantic_key
    assert default.semantic_key == explicit.semantic_key
    assert default.artifact_key == explicit.artifact_key
    assert default == explicit
    assert portable == explicit


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
            scan_operator=CxxOperator("::cuda::maximum<>", "int"),
        ),
    )

    assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert "require an initial value" in plan.unsupported.message
    assert "rank zero" in plan.unsupported.message


def test_partial_exclusive_sum_without_initial_value_is_unsupported():
    plan = _plan(
        this_warp(),
        _scan(
            mode="exclusive",
            valid_items=ArgumentBinding.static(5),
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
        (this_block(), "inclusive", "array", None),
        (
            this_block(),
            "inclusive",
            "scalar",
            CxxOperator("::cuda::maximum<>", "int"),
        ),
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
def test_default_exclusive_scan_with_initial_value_canonicalizes_to_plus(group):
    initial_value = CxxFunction("7", Dependency("T"), name="initial_value")
    implicit = _plan(
        group,
        _scan(
            dtype="int",
            mode="exclusive",
            initial_value=initial_value,
        ),
    )
    explicit = _plan(
        group,
        _scan(
            dtype="int",
            mode="exclusive",
            scan_operator=CxxOperator(
                "::cuda::std::plus<T>",
                Dependency("T"),
                name="scan_op",
            ),
            initial_value=initial_value,
        ),
    )

    assert implicit.target in {
        GroupLoweringTarget.CUB_BLOCK,
        GroupLoweringTarget.CUB_WARP,
    }
    assert implicit.implementation.method_name == "ExclusiveScan"
    assert [item.name for item in implicit.implementation.classify_method()] == [
        "temp_storage",
        "input",
        "output",
        "initial_value",
        "scan_op",
    ]
    assert implicit.call != explicit.call
    assert implicit.semantic_key == explicit.semantic_key
    assert implicit.artifact_key == explicit.artifact_key


@pytest.mark.parametrize("group", [this_block(), this_warp()])
def test_custom_exclusive_scan_with_initial_value_is_planned(group):
    operation = _scan(
        dtype="int",
        mode="exclusive",
        operand_kind="scalar",
        items_per_thread=1,
        scan_operator=CxxOperator("::cuda::maximum<>", "int"),
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
