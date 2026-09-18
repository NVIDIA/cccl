# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Shared Shuffle plans retain exact distances, ownership, and payload types."""

from dataclasses import replace

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cuda.coop._core import (
    BindingKind,
    LaunchFacts,
    StorageOwnership,
    this_block,
    this_warp,
)
from cuda.coop.cutlass._compiler import _state
from cuda.coop.cutlass._lowering import _shuffle

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _plan(**kwargs):
    options = dict(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(8, 3, 2)),
        dtype=cutlass.Int32,
        items_per_thread=None,
        mode="offset",
        distance=1,
    )
    options.update(kwargs)
    return _shuffle._make_shuffle_plan(**options)


@pytest.mark.parametrize("distance", (-2147483648, -17, 0, 1, 2147483647))
def test_exact_offset(distance):
    plan = _plan(distance=distance)
    assert plan.call.operation.primitive.distance.value == distance
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.participation.exact_block_dim == (8, 3, 2)


@pytest.mark.parametrize(
    "mode,distance",
    (("offset", -(1 << 31) - 1), ("offset", 1 << 31), ("rotate", 0), ("rotate", 48)),
)
def test_static_bounds(mode, distance):
    with pytest.raises(ValueError, match="distance"):
        _plan(mode=mode, distance=distance)


@pytest.mark.parametrize(
    "dtype",
    (
        cutlass.Int8,
        cutlass.Int16,
        cutlass.Int32,
        cutlass.Int64,
        cutlass.Uint8,
        cutlass.Uint16,
        cutlass.Uint32,
    ),
)
def test_runtime_distance(dtype):
    binding = _shuffle._distance_binding(dtype(1), array=False)
    assert binding.kind is BindingKind.RUNTIME


@pytest.mark.parametrize(
    "distance", (True, np.bool_(True), 1.5, cutlass.Float32(1), cutlass.Uint64(1))
)
def test_distance_type(distance):
    with pytest.raises(TypeError, match="distance"):
        _shuffle._distance_binding(distance, array=False)


@pytest.mark.parametrize("mode", ("up", "down"))
def test_array_contract(mode):
    plan = _plan(items_per_thread=3, mode=mode)
    assert plan.result.result_items_per_thread == 3
    assert plan.call.operation.primitive.distance.kind is BindingKind.OMITTED
    with pytest.raises(ValueError, match="exactly 1"):
        _plan(items_per_thread=3, mode=mode, distance=2)
    with pytest.raises(TypeError, match="compile-time 1"):
        _plan(items_per_thread=3, mode=mode, distance=cutlass.Int32(1))


def test_request_identity():
    a = _shuffle._CubShuffleRequest(_plan(distance=1), cutlass.Int32)
    b = _shuffle._CubShuffleRequest(_plan(distance=2), cutlass.Int32)
    assert a != b
    assert a.symbol_name != b.symbol_name
    with pytest.raises(ValueError, match="dtype"):
        _shuffle._CubShuffleRequest(a.plan, cutlass.Uint32)
    with pytest.raises(ValueError, match="implementation"):
        _shuffle._CubShuffleRequest(
            replace(
                a.plan,
                implementation=replace(
                    a.plan.implementation,
                    algorithm=replace(
                        a.plan.implementation.algorithm, method_name="Rotate"
                    ),
                ),
            ),
            cutlass.Int32,
        )


def test_group_restriction():
    with pytest.raises(NotImplementedError, match="block"):
        _plan(group=this_warp())


def test_failed_call_rollback(monkeypatch):
    snapshot, registered, restored = object(), [], []
    monkeypatch.setattr(_state, "snapshot_active_session_state", lambda: snapshot)
    monkeypatch.setattr(_state, "register_request", registered.append)
    monkeypatch.setattr(_state, "restore_active_session_state", restored.append)

    def fail(**kwargs):
        raise RuntimeError("Shuffle FFI failed")

    monkeypatch.setattr(_shuffle, "ffi", fail)
    with pytest.raises(RuntimeError, match="Shuffle FFI failed"):
        _shuffle.provider_shuffle(
            group=this_block(),
            launch=LaunchFacts(exact_block_dim=(32, 1, 1)),
            value=3,
            mode="rotate",
            distance=1,
        )
    assert len(registered) == 1
    assert restored == [snapshot]
