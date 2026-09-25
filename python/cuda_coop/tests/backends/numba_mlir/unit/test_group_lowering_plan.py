# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]

_STATIC_PROVENANCE_GLOBAL = np.int32(3)


def _planner(function, *, arg_types, block=(64, 1, 1)):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    func_ir = run_frontend(function)
    state = SimpleNamespace(func_ir=func_ir, args=arg_types)
    return _GroupCallPlanner(
        state,
        {"block": block, "grid": (1, 1, 1), "cluster": None},
    )


def _assigned_var(func_ir, name):
    from numba_cuda_mlir.numbair_transforms import ir

    return next(
        statement.target
        for block in func_ir.blocks.values()
        for statement in block.body
        if isinstance(statement, ir.Assign) and statement.target.name == name
    )


def test_group_planning_context_exposes_only_declared_operations():
    from cuda.coop.numba_mlir._compiler._group_planning import (
        GroupPlanningContext,
    )

    launch = object()
    planner = SimpleNamespace(
        launch=launch,
        _constant=lambda value: ("constant", value),
        _reject_literal_unroll_value=lambda *_args: None,
    )
    context = GroupPlanningContext(planner)

    assert context.launch is launch
    assert context.constant("value") == ("constant", "value")
    assert not hasattr(context, "_planner")
    assert not hasattr(context, "__dict__")
    with pytest.raises(AttributeError):
        context._planner = planner
    with pytest.raises(AttributeError):
        _ = context.state


def test_group_planner_tracks_runtime_scalar_expression_provenance():
    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop._core import BindingKind
    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    def provenance(source, scalar):
        alias = scalar
        index = cuda.threadIdx.x
        element = source[index]
        binary = alias + index
        unary = -alias
        cast = np.int32(binary)
        if index:
            merged = cast
        else:
            merged = element
        return merged, unary

    array_type = types.Array(types.int32, 1, "C")
    func_ir = run_frontend(provenance)
    planner = _GroupCallPlanner(
        SimpleNamespace(func_ir=func_ir, args=(array_type, types.int32)),
        {"block": (32, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )
    expected = {
        "scalar": types.int32,
        "alias": types.int32,
        "index": types.int32,
        "element": types.int32,
        "binary": types.int64,
        "unary": types.int64,
        "cast": types.int32,
        "merged": types.int32,
    }
    for name, dtype in expected.items():
        value = _assigned_var(func_ir, name)
        assert planner.context.dtype(value) == dtype
        assert planner.context.planning_binding(value).kind is BindingKind.RUNTIME


def test_group_planner_marks_only_explicit_static_scalar_provenance_static():
    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop._core import BindingKind
    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    free_scalar = np.int32(4)

    def provenance(literal):
        constant = 5
        global_value = _STATIC_PROVENANCE_GLOBAL
        free_value = free_scalar
        alias = constant
        if cuda.threadIdx.x:
            same_phi = 7
            different_phi = 8
        else:
            same_phi = 7
            different_phi = 9
        return (
            literal,
            constant,
            global_value,
            free_value,
            alias,
            same_phi,
            different_phi,
        )

    func_ir = run_frontend(provenance)
    planner = _GroupCallPlanner(
        SimpleNamespace(
            func_ir=func_ir,
            args=(types.IntegerLiteral(np.int32(1)),),
        ),
        {"block": (32, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )
    expected_static = {
        "literal": np.int32(1),
        "constant": 5,
        "global_value": np.int32(3),
        "free_value": np.int32(4),
        "alias": 5,
        "same_phi": 7,
    }
    for name, expected in expected_static.items():
        binding = planner.context.planning_binding(_assigned_var(func_ir, name))
        assert binding.kind is BindingKind.STATIC
        assert binding.value == expected
        assert type(binding.value) is type(expected)

    different = planner.context.planning_binding(
        _assigned_var(func_ir, "different_phi")
    )
    assert different.kind is BindingKind.RUNTIME


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
def test_runtime_arithmetic_controls_share_planning_and_rewrite_paths(qualified):
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop as root_coop
    from cuda.coop._core import BindingKind
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    module = numba_coop if qualified else root_coop

    def memory(source, control):
        index = cuda.threadIdx.x
        valid = control + index
        default = source[index]
        offset = np.int64(control)
        output = module.ThreadData(2, dtype=types.int32)
        return module.load(
            module.this_block(),
            source,
            output,
            valid_items=valid,
            oob_default=default,
            offset=offset,
        )

    array_type = types.Array(types.int32, 1, "C")
    arg_types = (array_type, types.int32)
    planner = _planner(memory, arg_types=arg_types, block=(32, 1, 1))
    assert planner.run()

    state = SimpleNamespace(
        func_ir=planner.func_ir,
        args=arg_types,
        typingctx=SimpleNamespace(refresh=lambda: None),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    for label in sorted(planner.func_ir.blocks):
        rewrite.match(
            planner.func_ir,
            planner.func_ir.blocks[label],
            state.typemap,
            state.calltypes,
        )
    assert len(rewrite._matches) == 1
    match = next(iter(rewrite._matches.values()))
    assert len(match.runtime_args) == 5
    for name in ("num_valid_items", "oob_default"):
        assert match.factory_kwargs[name].kind is BindingKind.RUNTIME
    assert rewrite._resolve_var_dtype(match.runtime_args[2]) == types.int64
    assert rewrite._resolve_var_dtype(match.runtime_args[3]) == types.int32
    assert rewrite._resolve_var_dtype(match.runtime_args[4]) == types.int64


def test_direct_load_provider_is_selected_from_complete_core_plan(monkeypatch):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop._core import (
        BindingKind,
        GroupLoweringTarget,
        StorageOwnership,
        SynchronizationScope,
    )
    from cuda.coop.numba_mlir._compiler import _group_load_store

    plans = []
    plan_group_primitive = _group_load_store.plan_group_primitive

    def capture_plan(call, launch):
        plan = plan_group_primitive(call, launch)
        plans.append(plan)
        return plan

    monkeypatch.setattr(
        _group_load_store,
        "plan_group_primitive",
        capture_plan,
    )

    def memory_with_storage(source):
        storage = numba_coop.TempStorage(
            256,
            alignment=16,
            sharing="exclusive",
        )
        output = numba_coop.ThreadData(2, dtype=types.int32)
        return numba_coop.load(
            numba_coop.this_block(),
            source,
            output,
            valid_items=31,
            oob_default=-1,
            temp_storage=storage,
        )

    def memory_without_storage(source):
        output = numba_coop.ThreadData(2, dtype=types.int32)
        return numba_coop.load(
            numba_coop.this_block(),
            source,
            output,
            valid_items=31,
            oob_default=-1,
        )

    array_type = types.Array(types.int32, 1, "C")
    for memory in (memory_with_storage, memory_without_storage):
        planner = _planner(memory, arg_types=(array_type,))
        assert planner.run()
    assert len(plans) == 2

    plan = plans[0]
    implicit_plan = plans[1]
    assert plan == implicit_plan
    assert plan.call.operation == implicit_plan.call.operation
    assert plan.semantic_key == implicit_plan.semantic_key
    assert plan.artifact_key == implicit_plan.artifact_key
    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.unsupported is None
    assert plan.artifact_key is not None
    assert plan.participation is not None
    assert plan.participation.exact_group_size == 64
    assert plan.participation.exact_block_dim == (64, 1, 1)
    assert plan.participation.uniform_arguments == (
        "valid_items",
        "oob_default",
    )
    assert plan.topology is not None
    assert plan.topology.thread_rank == "linear_thread_rank"
    assert plan.result is None
    assert not plan.call.operation.returns_value
    assert plan.synchronization is not None
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.NONE
    assert plan.temp_storage is not None
    assert plan.temp_storage.ownership is StorageOwnership.NONE
    assert plan.temp_storage.address_space is None
    assert plan.temp_storage.cpp_type is None
    assert plan.temp_storage.instances is None
    assert plan.temp_storage.instance_index is None
    assert not plan.temp_storage.exact_layout_required
    assert plan.temp_storage.sharing is None
    assert plan.temp_storage.requested_size_in_bytes is None
    assert plan.temp_storage.requested_alignment is None
    assert not plan.temp_storage.auto_sync
    assert plan.provenance is not None
    assert plan.provenance.semantic_key == (
        "CUB",
        "cub/block/block_load.cuh",
        "cub::BlockLoad",
        "Load",
    )
    semantics = plan.call.operation
    assert semantics.dtype == types.int32
    assert semantics.items_per_thread == 2
    assert semantics.valid_items.kind is BindingKind.STATIC
    assert semantics.valid_items.value == 31
    assert semantics.oob_default.kind is BindingKind.STATIC
    assert semantics.oob_default.value == -1


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
@pytest.mark.parametrize("operation", ["load", "store"])
def test_load_store_infer_untyped_payloads_symmetrically(
    monkeypatch,
    qualified,
    operation,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop as root_coop
    from cuda.coop.numba_mlir._compiler import _group_load_store

    module = numba_coop if qualified else root_coop
    plans = []
    plan_group_primitive = _group_load_store.plan_group_primitive

    def capture_plan(call, launch):
        plan = plan_group_primitive(call, launch)
        plans.append(plan)
        return plan

    monkeypatch.setattr(_group_load_store, "plan_group_primitive", capture_plan)

    if operation == "load":

        def memory(source, destination):
            output = module.ThreadData(2)
            return module.load(
                module.this_block(),
                source,
                output,
                algorithm="direct",
            )

    else:

        def memory(source, destination):
            output = module.ThreadData(2)
            output[0] = source[0]
            output[1] = source[1]
            module.store(
                module.this_block(),
                destination,
                output,
                algorithm="direct",
            )

    array_type = types.Array(types.int32, 1, "C")
    assert _planner(memory, arg_types=(array_type, array_type)).run()

    assert len(plans) == 1
    assert plans[0].call.operation.dtype == types.int32


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
@pytest.mark.parametrize("projection", ["alias", "tuple", "tuple-alias"])
def test_inferred_load_dtype_follows_output_aliases(qualified, projection):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop as root_coop

    module = numba_coop if qualified else root_coop

    if projection == "alias":

        def memory(source, flag):
            payload = module.ThreadData(2)
            alias = payload
            if flag:
                output = alias
            else:
                output = payload
            module.load(module.this_block(), source, output)
            exchanged = module.exchange(
                module.this_block(), payload, mode="blocked_to_striped"
            )
            return module.inclusive_sum(module.this_block(), exchanged[0])

    elif projection == "tuple":

        def memory(source, flag):
            payload = module.ThreadData(2)
            packed = (payload,)
            output = packed[0]
            module.load(module.this_block(), source, output)
            exchanged = module.exchange(
                module.this_block(), payload, mode="blocked_to_striped"
            )
            return module.inclusive_sum(module.this_block(), exchanged[0])

    else:

        def memory(source, flag):
            payload = module.ThreadData(2)
            packed = (payload,)
            if flag:
                alias = packed
            else:
                alias = (payload,)
            output = alias[-1]
            module.load(module.this_block(), source, output)
            exchanged = module.exchange(
                module.this_block(), payload, mode="blocked_to_striped"
            )
            return module.inclusive_sum(module.this_block(), exchanged[0])

    array_type = types.Array(types.int32, 1, "C")
    planner = _planner(memory, arg_types=(array_type, types.boolean))
    assert planner.run()
    assert (
        planner.context.dtype(_assigned_var(planner.func_ir, "payload")) == types.int32
    )


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
def test_inferred_load_dtype_rejects_conflicting_alias_writes(qualified):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop as root_coop
    from cuda.coop.numba_mlir._compiler._group_planner_support import GroupRewriteError

    module = numba_coop if qualified else root_coop

    def memory(source, conflicting, flag):
        payload = module.ThreadData(2)
        other = module.ThreadData(2)
        module.load(module.this_block(), source, payload)
        if flag:
            output = payload
        else:
            output = other
        module.load(module.this_block(), conflicting, output)

    array_type = types.Array(types.int32, 1, "C")
    conflicting_type = types.Array(types.float32, 1, "C")
    planner = _planner(memory, arg_types=(array_type, conflicting_type, types.boolean))
    with pytest.raises(GroupRewriteError, match="inconsistent dtypes"):
        planner.run()


def test_load_does_not_infer_dtype_for_an_earlier_exchange():
    from numba_cuda_mlir import types

    from cuda import coop
    from cuda.coop.numba_mlir._compiler._group_planner_support import GroupRewriteError

    def memory(source):
        payload = coop.ThreadData(2)
        exchanged = coop.exchange(coop.this_block(), payload, mode="blocked_to_striped")
        coop.load(coop.this_block(), source, payload)
        return exchanged

    array_type = types.Array(types.int32, 1, "C")
    with pytest.raises(GroupRewriteError, match="could not infer a dtype"):
        _planner(memory, arg_types=(array_type,)).run()


@pytest.mark.parametrize(
    ("metadata_overrides", "diagnostic"),
    [
        ({"storage_abi": "leading_pointer"}, "storage_abi"),
        ({"execution_scope": "warp"}, "execution_scope"),
        ({"synchronization_scope": "block"}, "synchronization_scope"),
        (
            {"execution_scope": "warp", "synchronization_scope": "warp"},
            "synchronization_scope",
        ),
    ],
)
def test_group_plan_rejects_incompatible_provider_metadata(
    monkeypatch,
    metadata_overrides,
    diagnostic,
):
    from dataclasses import replace

    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir import _lowering
    from cuda.coop.numba_mlir._compiler import _group_planning
    from cuda.coop.numba_mlir._compiler._group_planner_support import (
        GroupRewriteError,
    )
    from cuda.coop.numba_mlir._compiler._operations import factory_operation

    metadata = factory_operation(_lowering.load)
    assert metadata is not None
    metadata = replace(metadata, **metadata_overrides)
    monkeypatch.setattr(
        _group_planning,
        "factory_operation",
        lambda _factory: metadata,
    )

    def memory(source):
        output = numba_coop.ThreadData(2, dtype=types.int32)
        return numba_coop.load(numba_coop.this_block(), source, output)

    array_type = types.Array(types.int32, 1, "C")
    with pytest.raises(GroupRewriteError, match=diagnostic):
        _planner(memory, arg_types=(array_type,)).run()


def test_group_plan_allows_declared_sync_scope_when_auto_sync_is_disabled(
    monkeypatch,
):
    from dataclasses import replace

    from cuda.coop._core import (
        StorageOwnership,
        SynchronizationContract,
        SynchronizationScope,
        TempStorageContract,
        this_block,
    )
    from cuda.coop.numba_mlir._compiler import _group_planning
    from cuda.coop.numba_mlir._compiler._group_planning import (
        GroupPlanningContext,
    )
    from cuda.coop.numba_mlir._compiler._operations import (
        FactoryOperation,
        StorageABI,
    )
    from tests.support.group_planning import _load_store, _plan

    plan = _plan(this_block(), _load_store("load"))
    plan = replace(
        plan,
        synchronization=SynchronizationContract(
            converged_entry=True,
            storage_reuse_barrier=SynchronizationScope.NONE,
        ),
        temp_storage=TempStorageContract(
            ownership=StorageOwnership.CALLER,
            address_space="shared",
            cpp_type="TestStorage",
            instances=1,
            instance_index="cta",
            exact_layout_required=True,
            sharing="exclusive",
            auto_sync=False,
        ),
    )
    metadata = FactoryOperation(
        operation="test",
        namespace="test",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
    monkeypatch.setattr(
        _group_planning,
        "factory_operation",
        lambda _factory: metadata,
    )

    GroupPlanningContext._validate_provider_contract(plan, object())


def test_group_plan_allows_storage_free_group_execution(monkeypatch):
    from dataclasses import replace

    from cuda.coop._core import (
        GroupLoweringTarget,
        LaunchFacts,
        SynchronizationScope,
        make_group_primitive_call,
        resolve_thread_group,
        this_block,
    )
    from cuda.coop._core.group._contracts import _contracts
    from cuda.coop.numba_mlir._compiler import _group_planning
    from cuda.coop.numba_mlir._compiler._group_planning import (
        GroupPlanningContext,
    )
    from cuda.coop.numba_mlir._compiler._operations import (
        FactoryOperation,
        StorageABI,
    )
    from tests.support.group_planning import _load_store, _plan

    plan = _plan(this_block(), _load_store())
    launch = LaunchFacts(exact_block_dim=(64, 1, 1))
    resolved_group = resolve_thread_group(
        this_block().group_by(2), launch
    ).require_supported()
    topology, participation, synchronization, storage = _contracts(
        resolved_group,
        launch,
        result=None,
        storage_ownership=plan.temp_storage.ownership,
        cpp_type=plan.temp_storage.cpp_type,
    )
    plan = replace(
        plan,
        target=GroupLoweringTarget.CUDAX_GROUP,
        call=make_group_primitive_call(resolved_group, plan.call.operation),
        resolved_group=resolved_group,
        topology=topology,
        participation=participation,
        synchronization=synchronization,
        temp_storage=storage,
    )
    metadata = FactoryOperation(
        operation="test",
        namespace="test",
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.GROUP,
        synchronization_scope=SynchronizationScope.NONE,
    )
    monkeypatch.setattr(
        _group_planning,
        "factory_operation",
        lambda _factory: metadata,
    )

    GroupPlanningContext._validate_provider_contract(plan, object())


@pytest.mark.parametrize(
    "case",
    [
        "storage-bearing-plan",
        "planned-synchronization",
        "provider-storage",
        "provider-execution",
        "provider-synchronization",
    ],
)
def test_group_plan_preserves_other_group_execution_rejections(monkeypatch, case):
    from dataclasses import replace

    from cuda.coop._core import (
        GroupLoweringTarget,
        LaunchFacts,
        SynchronizationScope,
        make_group_primitive_call,
        resolve_thread_group,
        this_block,
    )
    from cuda.coop._core.group._contracts import _contracts
    from cuda.coop.numba_mlir._compiler import _group_planning
    from cuda.coop.numba_mlir._compiler._group_planner_support import (
        GroupRewriteError,
    )
    from cuda.coop.numba_mlir._compiler._group_planning import (
        GroupPlanningContext,
    )
    from cuda.coop.numba_mlir._compiler._operations import (
        FactoryOperation,
        StorageABI,
    )
    from tests.support.group_planning import _load_store, _plan

    storage_bearing = case == "storage-bearing-plan"
    plan = _plan(
        this_block(),
        _load_store(algorithm="transpose" if storage_bearing else "direct"),
    )
    planned_synchronization = (
        SynchronizationScope.GROUP
        if case in {"storage-bearing-plan", "planned-synchronization"}
        else SynchronizationScope.NONE
    )
    launch = LaunchFacts(exact_block_dim=(64, 1, 1))
    resolved_group = resolve_thread_group(
        this_block().group_by(2), launch
    ).require_supported()
    topology, participation, synchronization, storage = _contracts(
        resolved_group,
        launch,
        result=None,
        storage_ownership=plan.temp_storage.ownership,
        cpp_type=plan.temp_storage.cpp_type,
    )
    plan = replace(
        plan,
        target=GroupLoweringTarget.CUDAX_GROUP,
        call=make_group_primitive_call(resolved_group, plan.call.operation),
        resolved_group=resolved_group,
        topology=topology,
        participation=participation,
        synchronization=replace(
            synchronization,
            storage_reuse_barrier=planned_synchronization,
        ),
        temp_storage=storage,
    )
    provider_storage = (
        StorageABI.LEADING_POINTER
        if case in {"storage-bearing-plan", "provider-storage"}
        else StorageABI.NONE
    )
    provider_execution = (
        SynchronizationScope.BLOCK
        if case == "provider-execution"
        else SynchronizationScope.GROUP
    )
    provider_synchronization = (
        SynchronizationScope.GROUP
        if case
        in {
            "storage-bearing-plan",
            "planned-synchronization",
            "provider-synchronization",
        }
        else SynchronizationScope.NONE
    )
    metadata = FactoryOperation(
        operation="test",
        namespace="test",
        storage_abi=provider_storage,
        execution_scope=provider_execution,
        synchronization_scope=provider_synchronization,
    )
    monkeypatch.setattr(
        _group_planning,
        "factory_operation",
        lambda _factory: metadata,
    )

    with pytest.raises(
        GroupRewriteError,
        match="supported only for storage-free providers",
    ):
        GroupPlanningContext._validate_provider_contract(plan, object())


def test_group_plan_rejects_declared_sync_for_implementation_owned_no_sync(
    monkeypatch,
):
    from dataclasses import replace

    from cuda.coop._core import (
        StorageOwnership,
        SynchronizationContract,
        SynchronizationScope,
        TempStorageContract,
        this_block,
    )
    from cuda.coop.numba_mlir._compiler import _group_planning
    from cuda.coop.numba_mlir._compiler._group_planner_support import (
        GroupRewriteError,
    )
    from cuda.coop.numba_mlir._compiler._group_planning import (
        GroupPlanningContext,
    )
    from cuda.coop.numba_mlir._compiler._operations import (
        FactoryOperation,
        StorageABI,
    )
    from tests.support.group_planning import _load_store, _plan

    plan = _plan(this_block(), _load_store("load"))
    plan = replace(
        plan,
        synchronization=SynchronizationContract(
            converged_entry=True,
            storage_reuse_barrier=SynchronizationScope.NONE,
        ),
        temp_storage=TempStorageContract(
            ownership=StorageOwnership.IMPLEMENTATION,
            address_space="shared",
            cpp_type="TestStorage",
            instances=1,
            instance_index="cta",
            exact_layout_required=False,
            auto_sync=False,
        ),
    )
    metadata = FactoryOperation(
        operation="test",
        namespace="test",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
    monkeypatch.setattr(
        _group_planning,
        "factory_operation",
        lambda _factory: metadata,
    )

    with pytest.raises(GroupRewriteError, match="synchronization_scope"):
        GroupPlanningContext._validate_provider_contract(plan, object())


def test_logical_warp_plan_selects_typed_cub_provider(monkeypatch):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop._core import GroupLoweringTarget
    from cuda.coop.numba_mlir._compiler import _group_load_store

    plans = []
    plan_group_primitive = _group_load_store.plan_group_primitive

    def capture_plan(call, launch):
        plan = plan_group_primitive(call, launch)
        plans.append(plan)
        return plan

    monkeypatch.setattr(
        _group_load_store,
        "plan_group_primitive",
        capture_plan,
    )

    def memory(source):
        output = numba_coop.ThreadData(2, dtype=types.int32)
        return numba_coop.load(numba_coop.this_warp().group_by(8), source, output)

    array_type = types.Array(types.int32, 1, "C")
    planner = _planner(memory, arg_types=(array_type,))
    assert planner.run()

    assert len(plans) == 1
    plan = plans[0]
    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.unsupported is None
    assert plan.topology is not None
    assert plan.topology.group_kind == "threads_within_warp"
    assert plan.topology.logical_width == 8
    assert plan.topology.instances == 8
    assert plan.artifact_key is not None


@pytest.mark.parametrize(
    "oob_default",
    [True, np.float16(1), np.complex64(1 + 2j)],
    ids=["bool", "float16", "complex"],
)
def test_static_oob_default_rejects_before_provider_selection(monkeypatch, oob_default):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._compiler import _group_load_store

    def memory(source):
        output = numba_coop.ThreadData(2, dtype=types.int32)
        return numba_coop.load(
            numba_coop.this_block(),
            source,
            output,
            valid_items=1,
            oob_default=oob_default,
        )

    monkeypatch.setattr(
        _group_load_store._LoadStorePlanning,
        "_scope_factory",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid oob_default reached provider selection"
        ),
    )
    array_type = types.Array(types.int32, 1, "C")
    with pytest.raises((TypeError, ValueError), match="oob_default"):
        _planner(memory, arg_types=(array_type,)).run()


@pytest.mark.parametrize(
    "oob_type",
    [
        pytest.param("boolean", id="bool"),
        pytest.param("float16", id="float16"),
        pytest.param("complex64", id="complex"),
        pytest.param("optional", id="optional"),
        pytest.param("float32", id="mismatched"),
    ],
)
def test_runtime_oob_default_rejects_before_provider_selection(monkeypatch, oob_type):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._compiler import _group_load_store

    def memory(source, oob_default):
        output = numba_coop.ThreadData(2, dtype=types.int32)
        return numba_coop.load(
            numba_coop.this_block(),
            source,
            output,
            valid_items=1,
            oob_default=oob_default,
        )

    monkeypatch.setattr(
        _group_load_store._LoadStorePlanning,
        "_scope_factory",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid oob_default reached provider selection"
        ),
    )
    value_type = (
        types.Optional(types.int32)
        if oob_type == "optional"
        else getattr(types, oob_type)
    )
    array_type = types.Array(types.int32, 1, "C")
    with pytest.raises((TypeError, ValueError), match="oob_default|supports dtypes"):
        _planner(memory, arg_types=(array_type, value_type)).run()


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
@pytest.mark.parametrize("operation", ["load", "store"])
@pytest.mark.parametrize(
    "dtype_spelling",
    ["builtin", "string", "numpy-type", "numpy-dtype", "backend"],
)
def test_equivalent_dtype_spellings_are_canonicalized_before_planning(
    monkeypatch,
    qualified,
    operation,
    dtype_spelling,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop as root_coop
    from cuda.coop.numba_mlir._compiler import _group_load_store

    spellings = {
        "builtin": int,
        "string": "int32",
        "numpy-type": np.int32,
        "numpy-dtype": np.dtype(np.int32),
        "backend": types.int32,
    }
    dtype = spellings[dtype_spelling]
    module = numba_coop if qualified else root_coop
    plans = []
    plan_group_primitive = _group_load_store.plan_group_primitive

    def capture_plan(call, launch):
        plan = plan_group_primitive(call, launch)
        plans.append(plan)
        return plan

    monkeypatch.setattr(_group_load_store, "plan_group_primitive", capture_plan)

    if operation == "load":

        def memory(memory):
            payload = module.ThreadData(2, dtype=dtype)
            return module.load(module.this_block(), memory, payload)

    else:

        def memory(memory):
            payload = module.ThreadData(2, dtype=dtype)
            module.store(module.this_block(), memory, payload)

    array_type = types.Array(types.int32, 1, "C")
    planner = _planner(memory, arg_types=(array_type,))
    assert planner.run()
    assert len(plans) == 1
    assert plans[0].call.operation.dtype == types.int32


_STATIC_DEFAULT_DTYPES = (
    pytest.param("int8", np.int8, id="int8"),
    pytest.param("uint8", np.uint8, id="uint8"),
    pytest.param("int16", np.int16, id="int16"),
    pytest.param("uint16", np.uint16, id="uint16"),
    pytest.param("int32", np.int32, id="int32"),
    pytest.param("uint32", np.uint32, id="uint32"),
    pytest.param("int64", np.int64, id="int64"),
    pytest.param("uint64", np.uint64, id="uint64"),
    pytest.param("float32", np.float32, id="float32"),
    pytest.param("float64", np.float64, id="float64"),
)


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
@pytest.mark.parametrize(("dtype_name", "numpy_dtype"), _STATIC_DEFAULT_DTYPES)
def test_static_oob_default_boundaries_use_the_load_payload_dtype(
    qualified,
    dtype_name,
    numpy_dtype,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop as root_coop

    module = numba_coop if qualified else root_coop
    numpy_kind = np.dtype(numpy_dtype).kind
    info = np.iinfo(numpy_dtype) if numpy_kind in "iu" else np.finfo(numpy_dtype)
    boundaries = (int(info.min), int(info.max))
    if numpy_kind == "f":
        boundaries = (-float(info.max), float(info.max))

    for oob_default in boundaries:

        def memory(source):
            output = module.ThreadData(2, dtype=getattr(types, dtype_name))
            return module.load(
                module.this_block(),
                source,
                output,
                valid_items=1,
                oob_default=oob_default,
            )

        array_type = types.Array(getattr(types, dtype_name), 1, "C")
        assert _planner(memory, arg_types=(array_type,)).run()


@pytest.mark.parametrize("qualified", [False, True], ids=["root", "qualified"])
@pytest.mark.parametrize(
    ("value_type_name", "error"),
    [
        ("float32", "does not match payload dtype"),
        ("boolean", "supports dtypes"),
    ],
    ids=["mismatched", "unsupported"],
)
def test_untyped_store_infers_write_dtype_before_destination_fallback(
    monkeypatch,
    qualified,
    value_type_name,
    error,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop as root_coop
    from cuda.coop.numba_mlir._compiler import _group_load_store

    module = numba_coop if qualified else root_coop

    def memory(destination, value):
        payload = module.ThreadData(2)
        payload[0] = value
        payload[1] = value
        module.store(module.this_block(), destination, payload)

    array_type = types.Array(types.int32, 1, "C")
    planner = _planner(
        memory,
        arg_types=(array_type, getattr(types, value_type_name)),
    )
    monkeypatch.setattr(
        _group_load_store._LoadStorePlanning,
        "_scope_factory",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid Store writes reached provider selection"
        ),
    )

    with pytest.raises(TypeError, match=error):
        planner.run()


def test_scalar_scan_of_a_loaded_payload_element_plans_as_a_scalar(monkeypatch):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop._core import GroupOperandKind
    from cuda.coop.numba_mlir._compiler import _group_scan

    plans = []
    plan_group_primitive = _group_scan.plan_group_primitive

    def capture_plan(call, launch):
        plan = plan_group_primitive(call, launch)
        plans.append(plan)
        return plan

    monkeypatch.setattr(_group_scan, "plan_group_primitive", capture_plan)

    def kernel(source, output):
        payload = numba_coop.ThreadData(2, dtype=types.int32)
        numba_coop.load(
            numba_coop.this_block(),
            source,
            payload,
            algorithm="transpose",
        )
        # Indexing the loaded payload selects one scalar element.
        output[0] = numba_coop.inclusive_sum(numba_coop.this_block(), payload[0])

    array_type = types.Array(types.int32, 1, "C")
    planner = _planner(kernel, arg_types=(array_type, array_type))
    assert planner.run()

    assert len(plans) == 1
    assert plans[0].unsupported is None
    assert plans[0].result is not None
    assert plans[0].result.operand_kind is GroupOperandKind.SCALAR
