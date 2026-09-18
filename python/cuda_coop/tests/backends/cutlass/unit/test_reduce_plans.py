# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""CUTLASS Reduce consumes shared routing and result contracts."""

import operator
from dataclasses import replace

import pytest

cutlass = pytest.importorskip("cutlass")

from cuda.coop._core import (
    ArgumentBinding,
    GroupLoweringTarget,
    LaunchFactOrigin,
    LaunchFacts,
    ResultVisibility,
    StorageOwnership,
    SynchronizationScope,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)
from cuda.coop.cutlass._compiler import _rendering, _state
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES, INTEGER_VALUE_TYPES
from cuda.coop.cutlass._lowering import _reduce
from cuda.coop.cutlass._operators import normalize_operator, validate_operator_dtype

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _plan(group=None, **options):
    kwargs = dict(
        group=this_block() if group is None else group,
        launch=LaunchFacts(
            exact_block_dim=(8, 4, 2),
            exact_cluster_dim=(1, 1, 1),
            cluster_launch=False,
            provenance=LaunchFactOrigin("cluster_launch", "test", verified=True),
        ),
        dtype=cutlass.Int32,
        value_kind="scalar",
        items_per_thread=1,
        op="sum",
        broadcast=True,
    )
    kwargs.update(options)
    return _reduce._make_group_reduce_plan(**kwargs).require_supported()


@pytest.mark.parametrize(
    "group",
    (
        this_thread(),
        this_warp(),
        this_warp().group_by(8),
        this_block(),
        this_block().group_by(1),
        this_cluster(),
    ),
    ids=("thread", "warp", "logical", "block", "mapped", "cluster"),
)
@pytest.mark.parametrize(
    "op", ("sum", "multiplies", "min", "max", "bit_and", "bit_or", "bit_xor")
)
@pytest.mark.parametrize("broadcast", (True, False))
def test_cudax_route_and_visibility(group, op, broadcast):
    plan = _plan(
        group, op=op, broadcast=broadcast, value_kind="array", items_per_thread=3
    )
    request = _reduce._CudaxReduceRequest(plan, op, cutlass.Int32)
    assert plan.target is GroupLoweringTarget.CUDAX_GROUP
    assert plan.temp_storage.ownership is StorageOwnership.NONE
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.NONE
    expected = (
        ResultVisibility.ALL_MEMBERS if broadcast else ResultVisibility.GROUP_ROOT
    )
    assert plan.result.visibility is expected
    assert request.items_per_thread == 3


@pytest.mark.parametrize(
    "algorithm", ("raking_commutative_only", "raking", "warp_reductions")
)
def test_block_algorithm_selects_cub(algorithm):
    plan = _plan(
        algorithm=algorithm, broadcast=False, value_kind="array", items_per_thread=2
    )
    request = _reduce._CubReduceRequest(plan, "sum", cutlass.Int32)
    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.result.visibility is ResultVisibility.GROUP_ROOT
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK
    assert request.items_per_thread == 2


@pytest.mark.parametrize("width", (1, 2, 4, 8, 16, 32))
def test_partial_warp_plan(width):
    plan = _plan(
        this_warp().group_by(width),
        valid_items=ArgumentBinding.runtime(),
        broadcast=False,
    )
    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert _reduce._warp_instances(plan) == (64 // width, width)
    assert plan.result.visibility is ResultVisibility.GROUP_ROOT
    assert plan.participation.argument_preconditions[0].minimum == 1
    assert plan.participation.argument_preconditions[0].maximum == width


@pytest.mark.parametrize("dtype", tuple(ALL_PROVIDER_TYPES))
def test_builtin_dtype_profile(dtype):
    validate_operator_dtype("sum", dtype)
    if dtype in INTEGER_VALUE_TYPES:
        validate_operator_dtype("bit_xor", dtype)
    else:
        with pytest.raises(TypeError, match="integer dtype"):
            validate_operator_dtype("bit_xor", dtype)
    _reduce._CudaxReduceRequest(_plan(dtype=dtype), "sum", dtype)


@pytest.mark.parametrize(
    "value, expected",
    (
        (None, "sum"),
        (" PLUS ", "sum"),
        ("BIT-XOR", "bit_xor"),
        (operator.mul, "multiplies"),
    ),
)
def test_operator_aliases(value, expected):
    assert normalize_operator(value) == expected


def test_callback_rejected():
    with pytest.raises(NotImplementedError, match="custom callbacks"):
        normalize_operator(lambda a, b: a + b)

    class OperatorAlias:
        def __hash__(self):
            return hash(operator.add)

        def __eq__(self, other):
            return other is operator.add

        def __call__(self, a, b):
            return a - b

    with pytest.raises(NotImplementedError, match="custom callbacks"):
        normalize_operator(OperatorAlias())


@pytest.mark.parametrize("valid", (0, -1, 65))
def test_static_prefix_range(valid):
    with pytest.raises(ValueError, match="valid_items"):
        _plan(valid_items=ArgumentBinding.static(valid), broadcast=False)


def test_request_rejects_mismatched_plan():
    plan = _plan()
    with pytest.raises(ValueError, match="dtype"):
        _reduce._CudaxReduceRequest(plan, "sum", cutlass.Uint32)
    with pytest.raises(ValueError, match="operator"):
        _reduce._CudaxReduceRequest(plan, "max", cutlass.Int32)
    with pytest.raises(ValueError, match="result mode"):
        _reduce._CudaxReduceRequest(
            replace(
                plan, implementation=replace(plan.implementation, overload="root_only")
            ),
            "sum",
            cutlass.Int32,
        )


def test_grid_reduction_rejected():
    with pytest.raises(NotImplementedError, match="workspace"):
        _plan(
            this_grid(),
            launch=LaunchFacts(
                exact_block_dim=(64, 1, 1),
                exact_grid_dim=(2, 1, 1),
                exact_cluster_dim=(1, 1, 1),
                cluster_launch=False,
            ),
        )


def test_runtime_count_guard():
    plan = _plan(valid_items=ArgumentBinding.runtime(), broadcast=False)
    request = _reduce._CubReduceRequest(plan, "sum", cutlass.Int32)
    source = _rendering.render_bundle_source([request])
    assert 'asm volatile("trap;"' in source
    assert source.index('asm volatile("trap;"') < source.index(
        "implementation_type(storage).Sum"
    )


def test_failed_ffi_restores_session(monkeypatch):
    snapshot = object()
    registered, restored = [], []
    monkeypatch.setattr(_state, "snapshot_active_session_state", lambda: snapshot)
    monkeypatch.setattr(_state, "register_request", registered.append)
    monkeypatch.setattr(_state, "restore_active_session_state", restored.append)

    def fail_ffi(**kwargs):
        raise RuntimeError("reduction FFI failed")

    monkeypatch.setattr(_reduce, "ffi", fail_ffi)
    with pytest.raises(RuntimeError, match="reduction FFI failed"):
        _reduce.provider_reduce(
            group=this_block(), launch=LaunchFacts(exact_block_dim=(64, 1, 1)), value=3
        )
    assert len(registered) == 1
    assert restored == [snapshot]
