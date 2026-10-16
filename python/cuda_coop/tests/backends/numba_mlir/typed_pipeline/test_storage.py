# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile explicit scratch, reservations, and per-group storage views."""

from types import SimpleNamespace

import numba_cuda_mlir.tools as compiler_tools
import numpy as np
import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

from cuda import coop
from cuda.coop.numba_mlir._pipeline import _storage
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError

from ._support import requires_typed_pipeline

pytestmark = [
    requires_typed_pipeline,
    pytest.mark.backend_numba_mlir,
    pytest.mark.compile,
]


@pytest.fixture(autouse=True)
def _fixed_current_device(monkeypatch):
    monkeypatch.setattr(
        compiler_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )


def _compile(kernel, *args, block=(64, 1, 1)):
    return kernel._compile_launch_config_signature(
        types.void(*args),
        (
            ("grid", (1, 1, 1)),
            ("block", block),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )


@pytest.mark.parametrize("auto_sync", [False, True])
@pytest.mark.parametrize("sharing", ["shared", "exclusive"])
def test_explicit_scratch_composes_load_reduce_and_store(auto_sync, sharing):
    @cuda.jit(chip="sm_90")
    def kernel(source, output, items_per_thread):
        group = coop.this_block()
        scratch = coop.TempStorage(auto_sync=auto_sync, sharing=sharing)
        items = coop.ThreadData(items_per_thread)
        coop.load(
            group, source, items, algorithm="transpose", temp_storage=scratch
        )
        group.sync()
        total = coop.sum(group, items, temp_storage=scratch)
        group.sync()
        coop.store(
            group, output, items, algorithm="transpose", temp_storage=scratch
        )
        if group.rank() == 0:
            output[0] = total

    result = _compile(
        kernel, types.int32[::1], types.int32[::1], types.IntegerLiteral(4)
    )
    assert result.metadata["cubin"]


@pytest.mark.parametrize("auto_sync", [False, True])
def test_merged_storage_retains_constructor_identity(auto_sync):
    @cuda.jit(chip="sm_90")
    def kernel(source, output, choose):
        if choose:
            scratch = coop.TempStorage(auto_sync=auto_sync)
        else:
            scratch = coop.TempStorage(auto_sync=auto_sync)
        total = coop.sum(
            coop.this_block(), source[cuda.threadIdx.x], temp_storage=scratch
        )
        if cuda.threadIdx.x == 0:
            output[0] = total

    if auto_sync:
        assert _compile(
            kernel, types.int32[::1], types.int32[::1], types.boolean
        ).metadata["cubin"]
    else:
        with pytest.raises(TypingError, match="exactly one site"):
            _compile(kernel, types.int32[::1], types.int32[::1], types.boolean)


def test_exclusive_reservations_provide_typed_shared_arrays():
    @cuda.jit(chip="sm_90")
    def kernel(output):
        storage = coop.TempStorage(sharing="exclusive")
        first = storage.reserve(3, "uint8")
        reserve = storage.reserve
        second = reserve(4, types.float64, alignment=64)
        first[0] = 3
        second[0] = 7.0
        output[0] = first[0] + second[0] + first.size + second.size

    assert _compile(kernel, types.float64[::1]).metadata["cubin"]


def test_descriptor_tuple_cannot_escape_to_a_device_helper():
    @cuda.jit(device=True, inline="never")
    def take_second(pair):
        return pair[1]

    @cuda.jit(chip="sm_90")
    def kernel(output):
        storage = coop.TempStorage()
        values = storage.reserve(4, types.int32)
        values[0] = 7
        output[0] = take_second((storage, values[0]))

    with pytest.raises(TypedPipelineError, match="descriptor would escape"):
        _compile(kernel, types.int32[::1])


def test_numpy_integer_controls_preserve_their_ordinary_value_dtype():
    count = np.uint8(4)
    alignment = np.int64(64)

    @cuda.jit(chip="sm_90")
    def kernel(output):
        storage = coop.TempStorage(alignment=alignment)
        values = storage.reserve(count, types.int32, alignment=alignment)
        values[0] = 7
        item = coop.ThreadData(1)
        item[0] = count
        coop.store(coop.this_block(), output, item)

    # The captured scalar remains uint8 when it produces ordinary payload
    # data, even though its reserve count is resolved as a static integer.
    assert _compile(kernel, types.uint8[::1]).metadata["cubin"]


@pytest.mark.parametrize("auto_sync", [False, True])
@pytest.mark.parametrize("sharing", ["shared", "exclusive"])
def test_scan_respects_the_explicit_storage_contract(auto_sync, sharing):
    @cuda.jit(chip="sm_90")
    def kernel(source, output):
        scratch = coop.TempStorage(auto_sync=auto_sync, sharing=sharing)
        group = coop.this_block()
        first = coop.inclusive_sum(
            group, source[cuda.threadIdx.x], temp_storage=scratch
        )
        cuda.syncthreads()
        output[cuda.threadIdx.x] = coop.exclusive_sum(
            group, first, temp_storage=scratch
        )

    result = _compile(kernel, types.int32[::1], types.int32[::1])
    assert result.metadata["cubin"]
    assert result.metadata["mlir_module_str"].count("gpu.barrier") == (
        1 + 2 * auto_sync
    )


def test_logical_warp_scratch_uses_multidimensional_block_ranks():
    @cuda.jit(chip="sm_90")
    def kernel(source, output):
        group = coop.this_warp().group_by(8)
        thread = cuda.threadIdx.x + cuda.blockDim.x * cuda.threadIdx.y
        first = coop.sum(group, source[thread])
        second = coop.sum(group, source[thread] + 1)
        if group.rank() == 0:
            output[thread] = first + second

    assert _compile(
        kernel, types.int32[::1], types.int32[::1], block=(8, 8, 1)
    ).metadata["cubin"]


@pytest.fixture
def _shared_memory_limits(monkeypatch):
    monkeypatch.setattr(
        _storage,
        "_query_device_shared_memory_limits",
        lambda: {
            "max_default_shared_memory_per_block": 48 * 1024,
            "max_optin_shared_memory_per_block": 96 * 1024,
        },
    )


@pytest.mark.parametrize("reservation_only", [False, True])
def test_dynamic_backing_records_the_exact_launch_requirement(
    _shared_memory_limits, reservation_only
):
    @cuda.jit(chip="sm_90")
    def kernel(source, output):
        scratch = coop.TempStorage(64 * 1024)
        if reservation_only:
            tile = scratch.reserve(16 * 1024, types.int32)
            tile[cuda.threadIdx.x] = source[cuda.threadIdx.x]
            cuda.syncthreads()
            output[cuda.threadIdx.x] = tile[cuda.threadIdx.x] + tile.size
        else:
            total = coop.sum(
                coop.this_block(),
                source[cuda.threadIdx.x],
                temp_storage=scratch,
            )
            if cuda.threadIdx.x == 0:
                output[0] = total

    result = _compile(kernel, types.int32[::1], types.int32[::1])
    assert result.metadata["cubin"]
    assert result.metadata["required_dynamic_shared_memory"] == 64 * 1024


@pytest.mark.parametrize(
    "capacity,alignment,own_shared,message",
    [
        (64 * 1024, 64, False, "guarantees only 16-byte alignment"),
        (128 * 1024, 16, False, "device max opt-in is 98304 bytes"),
        (64 * 1024, 16, True, "static cuda.shared.array.*would alias"),
    ],
)
def test_dynamic_backing_checks_alignment_capacity_and_user_allocations(
    _shared_memory_limits, capacity, alignment, own_shared, message
):
    @cuda.jit(chip="sm_90")
    def kernel(output):
        scratch = coop.TempStorage(capacity, alignment=alignment)
        tile = scratch.reserve(1, types.int32)
        tile[0] = 1
        if own_shared:
            mine = cuda.shared.array(1, types.int32)
            mine[0] = 3
            tile[0] += mine[0]
        output[0] = tile[0]

    with pytest.raises(TypedPipelineError, match=message):
        _compile(kernel, types.int32[::1])
