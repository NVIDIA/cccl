# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What the typed pipeline refuses for load and store, and how it says so.

A refusal names the argument or the construct and is raised at compile
time; none falls back to the planner.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError

from .._support import BLOCK, CHIP, f32_in_out, requires_typed_pipeline

pytestmark = requires_typed_pipeline


def _compile(kernel, signature=None, block=BLOCK):
    return testing.compile_kernel(
        kernel, signature or f32_in_out(), block=block, chip=CHIP
    )


def test_raw_array_temp_storage_is_refused():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        scratch = cuda.shared.array(4096, types.uint8)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=scratch
        )
        coop.store(block, output, items)

    with pytest.raises(
        Exception, match="temp_storage must be a TempStorage descriptor"
    ):
        _compile(kernel)


def test_non_group_argument_is_refused():
    def kernel(source, output):
        items = coop.ThreadData(4)
        coop.load(1, source, items)
        coop.store(coop.this_block(), output, items)

    with pytest.raises(
        Exception, match=r"coop\.load: expected a cooperative thread group"
    ):
        _compile(kernel)


def test_the_common_api_refuses_a_local_array_load_payload():
    def kernel(source, output):
        items = cuda.local.array(4, types.float32)
        coop.load(coop.this_block(), source, items)
        output[0] = items[0]

    with pytest.raises(
        Exception,
        match=r"cuda\.coop\.load requires output to be a fixed-size "
        r"ThreadData payload in the common API; use cuda\.coop\.numba_mlir",
    ):
        _compile(kernel)


def test_the_common_api_refuses_a_local_array_store_payload():
    def kernel(source, output):
        items = cuda.local.array(4, types.float32)
        for i in range(4):
            items[i] = source[i]
        coop.store(coop.this_block(), output, items)

    with pytest.raises(
        Exception,
        match=r"cuda\.coop\.store accepts only a scalar or fixed-size "
        r"ThreadData value payload in the common API",
    ):
        _compile(kernel)


def test_a_qualified_array_payload_needs_a_local_array_constructor():
    def kernel(source, output):
        qualified_coop.store(qualified_coop.this_block(), output, source)

    with pytest.raises(
        TypedPipelineError,
        match=r"cuda\.coop\.numba_mlir\.store requires a static "
        r"items_per_thread extent before provider selection",
    ):
        _compile(kernel)


def test_qualified_local_array_payloads_must_agree_on_the_extent():
    def kernel(source, output, flag):
        if flag[0]:
            items = cuda.local.array(2, types.float32)
        else:
            items = cuda.local.array(4, types.float32)
        qualified_coop.load(qualified_coop.this_block(), source, items)
        output[0] = items[0]

    signature = types.void(*f32_in_out().args, types.int32[::1])
    with pytest.raises(
        TypedPipelineError,
        match=r"cuda\.coop\.numba_mlir loop-carried payloads have "
        r"inconsistent items_per_thread extents",
    ):
        _compile(kernel, signature)


def test_an_unknown_algorithm_lists_the_known_ones():
    def kernel(source, output):
        items = coop.ThreadData(4)
        coop.load(coop.this_block(), source, items, algorithm="diagonal")
        coop.store(coop.this_block(), output, items)

    with pytest.raises(
        Exception, match=r"unknown algorithm 'diagonal'; use one of .*direct"
    ):
        _compile(kernel)


def test_writes_that_disagree_without_coop_evidence():
    def kernel(source, output):
        items = coop.ThreadData(4)
        base = cuda.threadIdx.x * 4
        items[0] = 2.5
        for i in range(1, 4):
            items[i] = 1
        for i in range(4):
            output[base + i] = items[i]

    with pytest.raises(
        Exception,
        match=(
            r"Failed to infer a consistent dtype from coop\.ThreadData writes"
        ),
    ):
        _compile(kernel)


def test_a_read_before_a_producer_uses_the_producer_dtype():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        items[0] = 1
        first = items[0]
        coop.load(block, source, items)
        output[cuda.threadIdx.x] = first

    compiled = _compile(kernel)
    assert re.search(
        r"^\s*first = static_getitem.* :: float32$",
        compiled.typed_ir,
        re.MULTILINE,
    )


@pytest.mark.parametrize("through_loop", [False, True])
def test_a_join_preserves_each_allocations_write_contract(through_loop):
    def kernel(source, output, condition):
        first = coop.ThreadData(1)
        first[0] = source[0]
        second = coop.ThreadData(1)
        coop.load(coop.this_block(), source, second)
        if through_loop:
            selected = first
            for i in range(2):
                if condition:
                    selected = second
        else:
            selected = first if condition else second
        selected[0] = np.int64(16777217)
        output[cuda.threadIdx.x] = selected[0]

    with pytest.raises(
        Exception,
        match=(
            r"Failed to infer a consistent dtype from coop\.ThreadData writes"
        ),
    ):
        _compile(kernel, types.void(*f32_in_out().args, types.boolean))


def test_store_validates_the_dtype_inferred_from_writes():
    def kernel(source, output):
        items = coop.ThreadData(1)
        items[0] = np.float64(1.25)
        coop.store(coop.this_block(), output, items)

    with pytest.raises(
        Exception,
        match=r"payload dtype float64 conflicts with destination dtype float32",
    ):
        _compile(kernel)


def test_a_coop_call_in_a_separate_device_function():
    @cuda.jit(device=True, inline="never")
    def load_into(source, items):
        coop.load(coop.this_block(), source, items)

    def kernel(source, output):
        items = coop.ThreadData(4, dtype=source.dtype)
        load_into(source, items)
        coop.store(coop.this_block(), output, items)

    with pytest.raises(
        Exception,
        match=r"coop\.load at line \d+ is inside .*load_into.*, a separately "
        r"compiled device function",
    ):
        _compile(kernel)


def test_scratch_above_the_static_limit_requires_device_limits(monkeypatch):
    from cuda.coop.numba_mlir._pipeline import _storage

    def unavailable():
        raise RuntimeError("no current device")

    monkeypatch.setattr(
        _storage, "_query_device_shared_memory_limits", unavailable
    )

    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(8)
        coop.load(block, source, items, algorithm="transpose")
        coop.store(block, output, items)

    signature = types.void(types.float64[::1], types.float64[::1])
    with pytest.raises(
        TypedPipelineError,
        match=r"require an exact current-device shared-memory query",
    ):
        _compile(kernel, signature, block=(1024, 1, 1))


def test_scratch_counts_the_kernels_own_shared_memory():
    def kernel(source, output):
        block = coop.this_block()
        own = cuda.shared.array(48000, types.uint8)
        own[cuda.threadIdx.x] = 0
        items = coop.ThreadData(4)
        coop.load(block, source, items, algorithm="transpose")
        coop.store(block, output, items)

    with pytest.raises(
        TypedPipelineError, match=r"with the kernel's own 48000 bytes"
    ):
        _compile(kernel)


@pytest.mark.parametrize("runtime_shape", [False, True])
def test_scratch_rejects_user_dynamic_shared_memory(runtime_shape):
    def kernel(source, output):
        size = source[0] if runtime_shape else 0
        own = cuda.shared.array(size, types.int32)
        own[cuda.threadIdx.x] = 100000 + cuda.threadIdx.x
        cuda.syncthreads()
        items = coop.ThreadData(4)
        coop.load(coop.this_block(), source, items, algorithm="transpose")
        cuda.syncthreads()
        output[cuda.threadIdx.x] = own[cuda.threadIdx.x] + items[0]

    signature = types.void(types.int32[::1], types.int32[::1])
    with pytest.raises(
        TypedPipelineError, match=r"dynamic/runtime-sized .*would alias"
    ):
        _compile(kernel, signature)


def test_scratch_counts_multidimensional_user_shared_memory():
    def kernel(source, output):
        own = cuda.shared.array((128, 96), types.int32)
        own[cuda.threadIdx.x, 0] = source[cuda.threadIdx.x]
        items = coop.ThreadData(4)
        coop.load(coop.this_block(), source, items, algorithm="transpose")
        output[cuda.threadIdx.x] = own[cuda.threadIdx.x, 0] + items[0]

    signature = types.void(types.int32[::1], types.int32[::1])
    with pytest.raises(
        TypedPipelineError, match=r"with the kernel's own 49152 bytes"
    ):
        _compile(kernel, signature)


def test_scratch_counts_padding_between_user_shared_arrays():
    def kernel(source, output):
        first = cuda.shared.array(1, types.uint8)
        second = cuda.shared.array((120, 400), types.uint8, alignment=64)
        first[0] = 1
        second[cuda.threadIdx.x % 120, 0] = 2
        items = coop.ThreadData(4)
        coop.load(coop.this_block(), source, items, algorithm="transpose")
        output[cuda.threadIdx.x] = items[0] + first[0] + second[0, 0]

    with pytest.raises(
        TypedPipelineError, match=r"with the kernel's own 48064 bytes"
    ):
        _compile(kernel)
