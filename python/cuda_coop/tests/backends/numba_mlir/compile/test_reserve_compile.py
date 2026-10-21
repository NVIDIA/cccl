# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile invalid reservation uses through the public kernel API."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir._compiler._rewrite_support import (
    CoopSinglePhaseRewriteError,
)

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.fixture(autouse=True)
def _fixed_current_device(monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )


def _compile(kernel, *arg_types):
    return kernel._compile_launch_config_signature(
        types.void(*arg_types),
        (
            ("grid", (1, 1, 1)),
            ("block", (32, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )


@pytest.mark.parametrize("count", [0, -1, 1.5])
def test_reserve_rejects_invalid_element_count(count):
    @cuda.jit(chip="sm_90")
    def kernel(output):
        storage = coop.TempStorage()
        values = storage.reserve(count, types.int32)
        output[0] = values[0]

    with pytest.raises(
        (CoopSinglePhaseRewriteError, TypingError),
        match="reserve num_elems must be a compile-time positive integer",
    ):
        _compile(kernel, types.int32[::1])


def test_reserve_rejects_runtime_element_count():
    @cuda.jit(chip="sm_90")
    def kernel(output):
        storage = coop.TempStorage()
        values = storage.reserve(output[0], types.int32)
        output[0] = values[0]

    with pytest.raises(
        (CoopSinglePhaseRewriteError, TypingError),
        match="reserve num_elems must be a compile-time positive integer",
    ):
        _compile(kernel, types.int32[::1])


@pytest.mark.parametrize("alignment", [0, 3])
def test_reserve_rejects_invalid_alignment(alignment):
    @cuda.jit(chip="sm_90")
    def kernel(output):
        storage = coop.TempStorage()
        values = storage.reserve(4, types.int32, alignment=alignment)
        output[0] = values[0]

    with pytest.raises(
        (CoopSinglePhaseRewriteError, TypingError),
        match="reserve alignment must be a compile-time positive power of 2",
    ):
        _compile(kernel, types.int32[::1])


def test_reserve_capacity_includes_alignment_padding():
    @cuda.jit(chip="sm_90")
    def kernel(output):
        storage = coop.TempStorage(71, sharing="exclusive")
        prefix = storage.reserve(3, types.uint8)
        values = storage.reserve(1, types.float64, alignment=64)
        output[0] = prefix[0] + values[0]

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="TempStorage size_in_bytes is smaller than required",
    ):
        _compile(kernel, types.float64[::1])


def test_shared_reserve_capacity_uses_largest_requirement():
    @cuda.jit(chip="sm_90")
    def kernel(output):
        storage = coop.TempStorage(64)
        prefix = storage.reserve(3, types.uint8)
        values = storage.reserve(8, types.float64, alignment=64)
        output[0] = prefix.size + values.size

    _compile(kernel, types.float64[::1])


@pytest.mark.parametrize("access", ["direct", "helper", "bound_method"])
def test_reserve_rejects_auto_sync_for_its_descriptor(access):
    @cuda.jit(device=True, inline="always")
    def allocate(storage):
        alias = storage
        return alias.reserve(32, types.int32)

    @cuda.jit(chip="sm_90")
    def kernel(output):
        storage = coop.TempStorage(auto_sync=True)
        if access == "helper":
            values = allocate(storage)
        elif access == "bound_method":
            reserve = storage.reserve
            values = reserve(32, types.int32)
        else:
            values = storage.reserve(32, types.int32)
        output[cuda.threadIdx.x] = values[cuda.threadIdx.x]

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="TempStorage.reserve requires auto_sync=False",
    ):
        _compile(kernel, types.int32[::1])


@pytest.mark.parametrize(
    "dtype", [types.void, types.boolean], ids=["unsized", "boolean"]
)
def test_reserve_rejects_unsupported_dtype(dtype):
    @cuda.jit(chip="sm_90")
    def kernel(output):
        storage = coop.TempStorage()
        values = storage.reserve(4, dtype)
        output[0] = values[0]

    with pytest.raises(
        (CoopSinglePhaseRewriteError, TypingError),
        match=(
            "reserve dtype must be a compile-time fixed-size "
            "integer, floating-point, or complex dtype"
        ),
    ):
        _compile(kernel, types.int32[::1])


@pytest.mark.parametrize(
    "access", ["direct", "alias", "bound_method", "helper"]
)
def test_reserve_after_release_fails(access):
    @cuda.jit(device=True, inline="always")
    def allocate(storage):
        return storage.reserve(32, types.int32)

    @cuda.jit(chip="sm_90")
    def kernel(output):
        storage = coop.TempStorage()
        alias = storage
        reserve = storage.reserve
        initial = storage.reserve(32, types.int32)
        released = storage.release()
        output[0] = initial.size + released.size
        if access == "alias":
            values = alias.reserve(32, types.int32)
        elif access == "bound_method":
            values = reserve(32, types.int32)
        elif access == "helper":
            values = allocate(storage)
        else:
            values = storage.reserve(32, types.int32)
        output[0] = values.size

    with pytest.raises(
        CoopSinglePhaseRewriteError, match="after.*release|released"
    ):
        _compile(kernel, types.int32[::1])


def test_reserve_rejects_join_after_conditional_release():
    @cuda.jit(chip="sm_90")
    def kernel(output, release_now):
        storage = coop.TempStorage()
        initial = storage.reserve(32, types.int32)
        output[0] = initial.size
        if release_now:
            released = storage.release()
            output[0] = released.size
        values = storage.reserve(32, types.int32)
        output[0] = values.size

    with pytest.raises(
        CoopSinglePhaseRewriteError, match="after.*release|released"
    ):
        _compile(kernel, types.int32[::1], types.boolean)


def test_reserve_rejects_released_descriptor_on_loop_backedge():
    @cuda.jit(chip="sm_90")
    def kernel(output, iterations):
        storage = coop.TempStorage()
        for phase in range(iterations):
            values = storage.reserve(32, types.int32)
            released = storage.release()
            output[phase] = values.size + released.size

    with pytest.raises(
        CoopSinglePhaseRewriteError, match="after.*release|released"
    ):
        _compile(kernel, types.int32[::1], types.int32)


def test_reserve_accepts_fresh_descriptor_in_each_loop_iteration():
    @cuda.jit(chip="sm_90")
    def kernel(output, iterations):
        thread = cuda.threadIdx.x
        for phase in range(iterations):
            storage = coop.TempStorage()
            values = storage.reserve(32, types.int32)
            values[thread] = phase + thread
            cuda.syncthreads()
            released = storage.release(types.int32)
            output[phase * 32 + thread] = released[31 - thread]
            cuda.syncthreads()

    _compile(kernel, types.int32[::1], types.int32)


def test_reserve_accepts_release_in_mutually_exclusive_branches():
    @cuda.jit(chip="sm_90")
    def kernel(output, reverse):
        storage = coop.TempStorage()
        values = storage.reserve(32, types.int32)
        thread = cuda.threadIdx.x
        values[thread] = thread
        cuda.syncthreads()
        if reverse:
            reversed_values = storage.release(types.int32)
            output[thread] = reversed_values[31 - thread]
        else:
            forward_values = storage.release(types.int32)
            output[thread] = forward_values[thread]

    _compile(kernel, types.int32[::1], types.boolean)


def test_release_requires_reservation_extent_to_fit_dtype():
    @cuda.jit(chip="sm_90")
    def kernel(output):
        storage = coop.TempStorage()
        values = storage.reserve(3, types.uint8)
        output[0] = values.size
        released = storage.release(types.float64)
        output[0] = released.size

    with pytest.raises(CoopSinglePhaseRewriteError, match="not divisible"):
        _compile(kernel, types.int32[::1])
