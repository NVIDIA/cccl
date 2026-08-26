# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group load store planner contracts."""

from importlib import import_module

import numpy as np
import pytest

from tests.support.group_planning import (
    ArgumentBinding,
    ArgumentKind,
    GroupLoadStoreAlgorithm,
    GroupLoweringTarget,
    LaunchFacts,
    ParameterRole,
    PreconditionEnforcement,
    StorageOwnership,
    UnsupportedReasonCode,
    _load_store,
    _plan,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
    this_thread,
    this_warp,
)


class _ThreadData:
    def __init__(self, items_per_thread=2, *, dtype=np.int32, length=None):
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
    def __init__(self, items_per_thread=2, *, dtype=np.int32):
        self.items_per_thread = items_per_thread
        self.dtype = dtype
        self._items = [0] * items_per_thread

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class _TempStorage:
    size_in_bytes = 128
    alignment = 16
    auto_sync = True
    sharing = "shared"


class _CompilerDtype:
    name = "int32"


class _CompilerInteger:
    width = 32
    signed = True
    dtype = _CompilerDtype()

    def ir_value(self):
        return object()


def test_portable_load_store_validates_payloads_and_group_options(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.load_store")
    calls = []

    def marker(*args, **kwargs):
        calls.append((args, kwargs))
        return args[3] if args[0] == "load" else None

    monkeypatch.setattr(api, "_group_primitive_marker", marker)
    payload = _ThreadData()

    with dispatch._compiler_scope("test.backend"):
        assert api.load(this_block(), object(), payload) is payload
        untyped_output = _ThreadData(dtype=None)
        assert api.load(this_block(), object(), untyped_output) is untyped_output
        api.store(this_warp(), object(), np.int32(1))
        api.store(
            this_block(),
            object(),
            _ReadonlyThreadData(),
            temp_storage=_TempStorage(),
        )
        with pytest.raises(ValueError, match="oob_default requires valid_items"):
            api.load(this_block(), object(), payload, oob_default=0)
        with pytest.raises(ValueError, match="block-only"):
            api.load(
                this_warp(),
                object(),
                payload,
                algorithm="warp_transpose",
            )
        with pytest.raises(ValueError, match="only for blocks"):
            api.store(
                this_warp(),
                object(),
                np.int32(1),
                temp_storage=object(),
            )
        with pytest.raises(TypeError, match="must satisfy TempStorageLike"):
            api.load(
                this_block(),
                object(),
                payload,
                temp_storage=object(),
            )
        with pytest.raises(TypeError, match="fixed-size ThreadData"):
            api.load(this_block(), object(), object())
        with pytest.raises(TypeError, match="fixed-size ThreadData"):
            api.load(this_block(), object(), _ReadonlyThreadData())
        with pytest.raises(ValueError, match="must match the payload item count"):
            api.load(this_block(), object(), _ThreadData(length=1))
        with pytest.raises(TypeError, match="portable API"):
            api.load(this_block(), object(), _ThreadData(dtype=complex))
        with pytest.raises(TypeError, match="backend-specific payloads"):
            api.store(this_block(), object(), object())
        with pytest.raises(TypeError, match="group must be a ThreadGroup"):
            api.load(object(), object(), payload)
        with pytest.raises(NotImplementedError, match="group kind 'thread'"):
            api.load(this_thread(), object(), payload)

    assert len(calls) == 4


@pytest.mark.parametrize(
    ("operation", "payload", "kwargs", "exception", "message"),
    [
        ("load", _ThreadData(), {"valid_items": 1.5}, TypeError, "portable integer"),
        ("load", _ThreadData(), {"offset": "4"}, TypeError, "portable integer"),
        (
            "load",
            _ThreadData(),
            {"valid_items": 1, "oob_default": object()},
            TypeError,
            "portable numeric scalar",
        ),
        ("load", _ThreadData(), {"valid_items": -1}, ValueError, "between 0"),
        ("load", _ThreadData(), {"offset": -1}, ValueError, "between 0"),
        ("load", _ThreadData(), {"offset": 1 << 63}, ValueError, "between 0"),
        (
            "load",
            _ThreadData(),
            {"valid_items": 65},
            ValueError,
            "group tile size 64",
        ),
        (
            "store",
            np.int32(1),
            {"valid_items": 33},
            ValueError,
            "group tile size 32",
        ),
        (
            "store",
            _ReadonlyThreadData(),
            {"valid_items": 65},
            ValueError,
            "group tile size 64",
        ),
    ],
)
def test_portable_load_store_rejects_invalid_static_controls_before_delegation(
    monkeypatch,
    operation,
    payload,
    kwargs,
    exception,
    message,
):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.load_store")
    calls = []
    monkeypatch.setattr(
        api,
        "_group_primitive_marker",
        lambda *args, **marker_kwargs: calls.append((args, marker_kwargs)),
    )

    with dispatch._compiler_scope("test.backend"):
        with pytest.raises(exception, match=message):
            if operation == "load":
                api.load(this_warp(), object(), payload, **kwargs)
            else:
                api.store(this_warp(), object(), payload, **kwargs)

    assert calls == []


def test_portable_load_store_accepts_control_boundaries_and_compiler_integers(
    monkeypatch,
):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.load_store")
    calls = []

    def marker(*args, **kwargs):
        calls.append((args, kwargs))
        return args[3] if args[0] == "load" else None

    monkeypatch.setattr(api, "_group_primitive_marker", marker)
    output = _ThreadData()
    with dispatch._compiler_scope("test.backend"):
        assert (
            api.load(
                this_warp(),
                object(),
                output,
                valid_items=np.int32(64),
                oob_default=np.float32(0),
                offset=(1 << 63) - 1,
            )
            is output
        )
        api.store(
            this_warp(),
            object(),
            np.int32(1),
            valid_items=32,
            offset=_CompilerInteger(),
        )
        api.store(
            this_warp(),
            object(),
            _ReadonlyThreadData(),
            valid_items=64,
        )

    assert len(calls) == 3


@pytest.mark.parametrize(
    ("group", "kind", "target", "cpp_class"),
    [
        (this_block(), "load", GroupLoweringTarget.CUB_BLOCK, "cub::BlockLoad"),
        (this_block(), "store", GroupLoweringTarget.CUB_BLOCK, "cub::BlockStore"),
        (this_warp(), "load", GroupLoweringTarget.CUB_WARP, "cub::WarpLoad"),
        (this_warp(), "store", GroupLoweringTarget.CUB_WARP, "cub::WarpStore"),
    ],
)
def test_group_load_store_selects_real_cub(group, kind, target, cpp_class):
    plan = _plan(group, _load_store(kind, items_per_thread=3), 64)

    assert plan.target is target
    assert plan.provenance.library == "CUB"
    assert plan.provenance.cpp_class == cpp_class
    if kind == "load":
        assert plan.result.result_items_per_thread == 3
    else:
        assert plan.result is None


def test_group_load_models_partial_tile_and_offset_bindings():
    operation = _load_store(
        valid_items=ArgumentBinding.runtime(),
        oob_default=ArgumentBinding.static(0),
        offset=ArgumentBinding.static(4),
    )
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=64))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert (
        plan.participation.valid_member_selection == "first valid_items tile elements"
    )
    assert plan.participation.uniform_arguments == (
        "valid_items",
        "oob_default",
        "offset",
    )
    assert [
        (classification.name, classification.kind, classification.role)
        for classification in call.argument_classifications
    ] == [
        ("source", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("output", ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        ("valid_items", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("oob_default", ArgumentKind.STATIC, ParameterRole.CONSTANT),
        ("offset", ArgumentKind.STATIC, ParameterRole.CONSTANT),
        ("algorithm", ArgumentKind.STATIC, ParameterRole.CONSTANT),
    ]
    assert len(plan.implementation.parameters) == 1
    implementation_controls = plan.implementation.classify_method()[-3:]
    assert [item.kind for item in implementation_controls] == [
        ArgumentKind.RUNTIME,
        ArgumentKind.STATIC,
        ArgumentKind.STATIC,
    ]
    precondition = plan.participation.argument_preconditions[0]
    assert (precondition.minimum, precondition.maximum) == (0, 128)
    assert precondition.enforcement is PreconditionEnforcement.CALLER


@pytest.mark.parametrize(
    ("group", "launch", "maximum"),
    [
        (this_block(), 64, 128),
        (this_warp(), 64, 64),
        (this_warp().group_by(8), 64, 16),
    ],
)
def test_group_load_store_valid_items_range(group, launch, maximum):
    for value in (np.int32(0), np.int64(maximum)):
        plan = _plan(
            group,
            _load_store(valid_items=ArgumentBinding.static(value)),
            launch,
        )
        precondition = plan.participation.argument_preconditions[0]
        assert (precondition.minimum, precondition.maximum) == (0, maximum)
        assert precondition.enforcement is PreconditionEnforcement.PLANNER_VALIDATED

    for value in (-1, maximum + 1):
        with pytest.raises(ValueError, match="group tile size"):
            _plan(
                group,
                _load_store(valid_items=ArgumentBinding.static(value)),
                launch,
            )

    with pytest.raises(TypeError, match="must be an integer"):
        _plan(
            group,
            _load_store(valid_items=ArgumentBinding.static(1.5)),
            launch,
        )


@pytest.mark.parametrize("group", [this_block(), this_warp()])
def test_group_load_store_canonicalizes_static_valid_items_identity(group):
    plain_operation = _load_store(valid_items=ArgumentBinding.static(5))
    numpy_operation = _load_store(valid_items=ArgumentBinding.static(np.int32(5)))
    plain_call = make_group_primitive_call(group, plain_operation)
    numpy_call = make_group_primitive_call(group, numpy_operation)
    plain_plan = plan_group_primitive(plain_call, LaunchFacts(exact_block_dim=64))
    numpy_plan = plan_group_primitive(numpy_call, LaunchFacts(exact_block_dim=64))

    assert numpy_operation.valid_items == ArgumentBinding.static(5)
    assert plain_operation.semantic_key == numpy_operation.semantic_key
    assert plain_call.semantic_key == numpy_call.semantic_key
    assert plain_plan.semantic_key == numpy_plan.semantic_key
    assert plain_plan.artifact_key == numpy_plan.artifact_key


@pytest.mark.parametrize("group", [this_block(), this_warp()])
def test_group_load_store_canonicalizes_static_offset_identity(group):
    plain_operation = _load_store(offset=ArgumentBinding.static(5))
    numpy_operation = _load_store(offset=ArgumentBinding.static(np.int64(5)))
    plain_call = make_group_primitive_call(group, plain_operation)
    numpy_call = make_group_primitive_call(group, numpy_operation)
    plain_plan = plan_group_primitive(plain_call, LaunchFacts(exact_block_dim=64))
    numpy_plan = plan_group_primitive(numpy_call, LaunchFacts(exact_block_dim=64))

    assert numpy_operation.offset == ArgumentBinding.static(5)
    assert plain_operation.semantic_key == numpy_operation.semantic_key
    assert plain_call.semantic_key == numpy_call.semantic_key
    assert plain_plan.semantic_key == numpy_plan.semantic_key
    assert plain_plan.artifact_key == numpy_plan.artifact_key


@pytest.mark.parametrize("kind", ["load", "store"])
@pytest.mark.parametrize(
    ("offset_binding", "argument_kind", "static_value"),
    [
        (ArgumentBinding.static(4), ArgumentKind.STATIC, 4),
        (ArgumentBinding.runtime(), ArgumentKind.RUNTIME, None),
    ],
)
def test_warp_load_store_preserves_pointer_offset_in_the_planned_abi(
    kind,
    offset_binding,
    argument_kind,
    static_value,
):
    operation = _load_store(
        kind,
        offset=offset_binding,
    )
    plan = _plan(this_warp(), operation, 64)

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert len(plan.implementation.parameters) == 1
    offset = plan.implementation.parameters[0][-1]
    assert offset.name == "offset"
    assert offset.argument_kind is argument_kind
    assert offset.static_value == static_value
    assert plan.implementation.metadata["pointer_offset"] == (
        offset_binding.semantic_key
    )


def test_group_load_store_supports_logical_warps_and_rejects_invalid_algorithms():
    mapped = this_warp().group_by(8)
    mapped_plan = _plan(mapped, _load_store(), 64)
    warp_plan = _plan(
        this_warp(),
        _load_store(algorithm=GroupLoadStoreAlgorithm.WARP_TRANSPOSE),
        64,
    )

    assert mapped_plan.target is GroupLoweringTarget.CUB_WARP
    assert mapped_plan.implementation.template_arguments["LOGICAL_WARP_THREADS"] == 8
    assert mapped_plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert mapped_plan.temp_storage.instances is None
    assert warp_plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT


@pytest.mark.parametrize("kind", ["load", "store"])
@pytest.mark.parametrize(
    "algorithm",
    [
        GroupLoadStoreAlgorithm.WARP_TRANSPOSE,
        GroupLoadStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
    ],
)
def test_block_warp_transpose_requires_complete_physical_warps(kind, algorithm):
    unsupported = _plan(
        this_block(),
        _load_store(kind, algorithm=algorithm),
        (16, 3, 1),
    )
    supported = _plan(
        this_block(),
        _load_store(kind, algorithm=algorithm),
        (16, 2, 1),
    )

    assert unsupported.target is GroupLoweringTarget.UNSUPPORTED
    assert unsupported.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert supported.target is GroupLoweringTarget.CUB_BLOCK
