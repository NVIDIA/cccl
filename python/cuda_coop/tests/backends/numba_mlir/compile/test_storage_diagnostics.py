# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Compile MVP descriptor restrictions through the production pipeline."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.numba_cuda.misc.special import literal_unroll

import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir._compiler._group_planner_support import GroupRewriteError
from cuda.coop.numba_mlir._compiler._rewrite_support import CoopSinglePhaseRewriteError

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


def _compile(kernel, *arg_types, block=(32, 1, 1)):
    return kernel._compile_launch_config_signature(
        types.void(*arg_types),
        (
            ("grid", (1, 1, 1)),
            ("block", block),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )


@pytest.mark.parametrize("qualified", (False, True), ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", (None, types.int32), ids=("inferred", "explicit"))
def test_load_return_cannot_be_used_as_a_store_payload(qualified, dtype):
    from cuda import coop as common_coop

    module = coop if qualified else common_coop

    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        payload = module.ThreadData(2, dtype)
        result = module.load(module.this_block(), source, payload)
        module.store(module.this_block(), destination, result)

    with pytest.raises(TypingError, match="none|None"):
        _compile(kernel, types.int32[::1], types.int32[::1])


def test_default_helper_accepts_descriptor_alias_chain():
    @cuda.jit(device=True)
    def helper(source, storage):
        payload = coop.ThreadData(1, types.int32)
        coop.load(coop.this_block(), source, payload, temp_storage=storage)
        return payload[0]

    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        storage = coop.TempStorage()
        alias = storage
        chained = alias
        destination[cuda.threadIdx.x] = helper(source, chained)

    assert _compile(kernel, types.int32[::1], types.int32[::1]).metadata["ltoir"]


def test_none_alias_reports_descriptor_error_before_type_inference():
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        empty = None
        storage = coop.TempStorage()
        if source[0] > 0:
            storage = empty
        payload = coop.ThreadData(1, types.int32)
        coop.load(coop.this_block(), source, payload, temp_storage=storage)
        destination[cuda.threadIdx.x] = payload[0]

    with pytest.raises(
        (GroupRewriteError, TypingError), match="TempStorage.*on every path"
    ):
        _compile(kernel, types.int32[::1], types.int32[::1])


def test_standalone_primitive_helper_reports_inline_requirement():
    @cuda.jit(device=True, inline="never")
    def standalone(destination, value):
        coop.store(coop.this_block(), destination, value)

    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        standalone(destination, source[cuda.threadIdx.x])

    with pytest.raises(
        (GroupRewriteError, TypingError), match="standalone.*must be inlined"
    ):
        _compile(kernel, types.int32[::1], types.int32[::1])


@pytest.mark.parametrize(
    "payload_kind", ["thread_data", "local_array", "local_array_keyword"]
)
def test_literal_unroll_cannot_determine_cooperative_payload_shape(payload_kind):
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        for count in literal_unroll((1, 2)):
            if payload_kind == "thread_data":
                payload = coop.ThreadData(count, types.int32)
            elif payload_kind == "local_array":
                payload = cuda.local.array(count, types.int32)
            else:
                payload = cuda.local.array(shape=count, dtype=types.int32)
            coop.load(coop.this_block(), source, payload)
            destination[cuda.threadIdx.x] = payload[0]

    with pytest.raises(
        (GroupRewriteError, TypingError), match="does not support literal_unroll values"
    ):
        _compile(kernel, types.int32[::1], types.int32[::1])


def test_unrelated_literal_unroll_can_coexist_with_cooperative_calls():
    @cuda.jit(chip="sm_90")
    def kernel(destination):
        value = 0
        for item in literal_unroll((1, 2)):
            value += item
        coop.store(coop.this_block(), destination, value)

    assert _compile(kernel, types.int64[::1]).metadata["ltoir"]


def test_storage_free_operation_can_coexist_with_user_dynamic_shared_memory():
    @cuda.jit(chip="sm_90")
    def kernel(destination):
        mine = cuda.shared.array(0, types.int32)
        mine[cuda.threadIdx.x] = 42
        coop.store(coop.this_block(), destination, mine[cuda.threadIdx.x])

    assert _compile(kernel, types.int32[::1]).metadata["ltoir"]


def test_literal_unroll_cannot_determine_cooperative_selector():
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        for algorithm in literal_unroll(("direct", "striped")):
            coop.store(
                coop.this_block(),
                destination,
                source[cuda.threadIdx.x],
                algorithm=algorithm,
            )

    with pytest.raises(
        (GroupRewriteError, TypingError), match="does not support literal_unroll values"
    ):
        _compile(kernel, types.int32[::1], types.int32[::1])


@pytest.mark.parametrize("dynamic_backing", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("user_shape", ["static", "zero", "runtime"])
def test_inlined_user_shared_allocation_is_checked_after_inlining(
    monkeypatch, dynamic_backing, user_shape
):
    from cuda.coop.numba_mlir._compiler import _rewrite_storage

    monkeypatch.setattr(
        _rewrite_storage,
        "_query_device_shared_memory_limits",
        lambda: {
            "max_default_shared_memory_per_block": 48 * 1024,
            "max_optin_shared_memory_per_block": 96 * 1024,
        },
    )
    size_in_bytes = 64 * 1024 if dynamic_backing else None
    shape = 32 if user_shape == "static" else 0

    if user_shape == "runtime":

        @cuda.jit(device=True)
        def user_allocation(count):
            allocate = cuda.shared.array
            alias = allocate
            return alias(count, types.int32)

    else:

        @cuda.jit(device=True)
        def user_allocation(count):
            allocate = cuda.shared.array
            alias = allocate
            return alias(shape, types.int32)

    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        mine = user_allocation(source[0])
        thread = cuda.threadIdx.x
        mine[thread] = source[thread]
        storage = coop.TempStorage(size_in_bytes)
        payload = coop.ThreadData(1, types.int32)
        payload[0] = mine[thread]
        coop.store(
            coop.this_block(),
            destination,
            payload,
            algorithm="transpose",
            temp_storage=storage,
        )

    if not dynamic_backing and user_shape == "static":
        assert _compile(kernel, types.int32[::1], types.int32[::1]).metadata["ltoir"]
    else:
        with pytest.raises(
            CoopSinglePhaseRewriteError,
            match=r"shared-memory backing.*test_storage_diagnostics.py.*would alias",
        ):
            _compile(kernel, types.int32[::1], types.int32[::1])


@pytest.mark.parametrize("argument", ["storage", "group_by"])
def test_literal_unroll_cannot_determine_cooperative_storage_or_partition(argument):
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        for count in literal_unroll((32, 64)):
            block = coop.this_block()
            if argument == "storage":
                storage = coop.TempStorage(count)
                payload = coop.ThreadData(1, types.int32)
                payload[0] = source[cuda.threadIdx.x]
                coop.store(
                    block,
                    destination,
                    payload,
                    algorithm="transpose",
                    temp_storage=storage,
                )
            else:
                group = block.group_by(count)
                coop.store(group, destination, source[cuda.threadIdx.x])

    with pytest.raises(
        (GroupRewriteError, TypingError), match="does not support literal_unroll values"
    ):
        _compile(kernel, types.int32[::1], types.int32[::1])


def test_implicit_oversized_storage_rejects_user_static_shared_allocation(monkeypatch):
    from cuda.coop.numba_mlir._compiler import _rewrite_storage

    monkeypatch.setattr(
        _rewrite_storage,
        "_query_device_shared_memory_limits",
        lambda: {
            "max_default_shared_memory_per_block": 48 * 1024,
            "max_optin_shared_memory_per_block": 96 * 1024,
        },
    )

    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        mine = cuda.shared.array(1024, types.int32)
        thread = cuda.threadIdx.x
        mine[thread] = source[thread]
        payload = coop.ThreadData(16, types.int32)
        for item in range(16):
            payload[item] = mine[thread]
        coop.store(coop.this_block(), destination, payload, algorithm="transpose")

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="dynamic shared-memory backing.*static cuda.shared.array.*would alias",
    ):
        _compile(kernel, types.int32[::1], types.int32[::1], block=(1024, 1, 1))
