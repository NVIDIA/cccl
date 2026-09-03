# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _planner(function, *, arg_types, block=(64, 1, 1)):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    func_ir = run_frontend(function)
    state = SimpleNamespace(func_ir=func_ir, args=arg_types)
    return _GroupCallPlanner(
        state,
        {"block": block, "grid": (1, 1, 1), "cluster": None},
    )


def test_direct_load_provider_is_selected_from_complete_core_plan(monkeypatch):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop._core import (
        BindingKind,
        GroupLoweringTarget,
        GroupOperandKind,
        ResultVisibility,
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

    def memory(source):
        storage = coop.TempStorage(
            256,
            alignment=16,
            sharing="exclusive",
        )
        output = coop.ThreadData(2, dtype=types.int32)
        return coop.load(
            coop.this_block(),
            source,
            output,
            valid_items=31,
            oob_default=-1,
            temp_storage=storage,
        )

    array_type = types.Array(types.int32, 1, "C")
    planner = _planner(memory, arg_types=(array_type,))
    assert planner.run()
    assert len(plans) == 1

    plan = plans[0]
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
    assert plan.result is not None
    assert plan.result.visibility is ResultVisibility.PER_MEMBER
    assert plan.result.operand_kind is GroupOperandKind.ARRAY
    assert plan.result.result_items_per_thread == 2
    assert plan.synchronization is not None
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.NONE
    assert plan.temp_storage is not None
    assert plan.temp_storage.ownership is StorageOwnership.CALLER
    assert plan.temp_storage.sharing == "exclusive"
    assert plan.temp_storage.requested_size_in_bytes == 256
    assert plan.temp_storage.requested_alignment == 16
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


def test_nonblock_plan_is_typed_before_provider_selection(monkeypatch):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop._core import GroupLoweringTarget, UnsupportedReasonCode
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
        output = coop.ThreadData(2, dtype=types.int32)
        return coop.load(coop.this_warp(), source, output)

    array_type = types.Array(types.int32, 1, "C")
    planner = _planner(memory, arg_types=(array_type,))
    monkeypatch.setattr(
        _group_load_store._LoadStorePlanning,
        "_scope_factory",
        lambda *_args, **_kwargs: pytest.fail(
            "unsupported plan reached provider selection"
        ),
    )
    with pytest.raises(NotImplementedError, match="this_block"):
        planner.run()

    assert len(plans) == 1
    plan = plans[0]
    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported is not None
    assert plan.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert plan.artifact_key is None


@pytest.mark.parametrize(
    "oob_default",
    [True, np.float16(1), np.complex64(1 + 2j)],
    ids=["bool", "float16", "complex"],
)
def test_static_oob_default_rejects_before_provider_selection(monkeypatch, oob_default):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler import _group_load_store

    def memory(source):
        output = coop.ThreadData(2, dtype=types.int32)
        return coop.load(
            coop.this_block(),
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

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler import _group_load_store

    def memory(source, oob_default):
        output = coop.ThreadData(2, dtype=types.int32)
        return coop.load(
            coop.this_block(),
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

    import cuda.coop.numba_mlir as qualified_coop
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
    module = qualified_coop if qualified else root_coop
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

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop
    from cuda.coop.numba_mlir._compiler import _group_load_store

    module = qualified_coop if qualified else root_coop

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
