# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.numba_cuda.misc.special import literal_unroll

import cuda.coop.numba_mlir as numba_coop
from cuda.coop.numba_mlir._compiler._group_planner_support import (
    GroupRewriteError,
)
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


def _compile(kernel, *arg_types, block=(32, 1, 1), cluster=None):
    return kernel._compile_launch_config_signature(
        types.void(*arg_types),
        (
            ("grid", (1, 1, 1)),
            ("block", block),
            ("sharedmem", 0),
            ("cluster", cluster),
        ),
    )


@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
@pytest.mark.parametrize(
    "dtype", (None, types.int32), ids=("inferred", "explicit")
)
def test_load_return_cannot_be_used_as_a_store_payload(qualified, dtype):
    from cuda import coop as common_coop

    module = numba_coop if qualified else common_coop

    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        payload = module.ThreadData(2, dtype)
        result = module.load(module.this_block(), source, payload)
        module.store(module.this_block(), destination, result)

    with pytest.raises(TypingError, match="none|None"):
        _compile(kernel, types.int32[::1], types.int32[::1])


def test_rebound_explicit_dtype_is_not_replaced_by_write_inference():
    @cuda.jit(chip="sm_90")
    def kernel(destination):
        if destination[0] > 0:
            dtype = types.int32
            first = coop.ThreadData(2, dtype)
            first[0] = 16777217
            destination[0] = first[0]
            dtype = types.float32
            second = coop.ThreadData(2, dtype)
            second[0] = 1.5
            destination[1] = second[0]

    with pytest.raises(
        (CoopSinglePhaseRewriteError, TypingError),
        match="dtype must resolve",
    ):
        _compile(kernel, types.int32[::1])


def test_rebound_shared_shape_is_rejected_with_cooperative_storage():
    @cuda.jit(chip="sm_90")
    def kernel(destination):
        for _ in range(2):
            size = 0
            dynamic = cuda.shared.array(size, types.int32)
            size = 32
            static = cuda.shared.array(size, types.int32)
            dynamic[cuda.threadIdx.x] = 1
            static[cuda.threadIdx.x] = 2
            payload = coop.ThreadData(2, types.int32)
            payload[0] = dynamic[cuda.threadIdx.x]
            payload[1] = static[cuda.threadIdx.x]
            coop.store(
                coop.this_block(), destination, payload, algorithm="transpose"
            )

    with pytest.raises(
        (CoopSinglePhaseRewriteError, TypingError), match="would alias"
    ):
        _compile(kernel, types.int32[::1])


def test_default_helper_accepts_descriptor_alias_chain():
    @cuda.jit(device=True)
    def helper(source, storage):
        payload = numba_coop.ThreadData(1, types.int32)
        numba_coop.load(
            numba_coop.this_block(), source, payload, temp_storage=storage
        )
        return payload[0]

    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        storage = numba_coop.TempStorage()
        alias = storage
        chained = alias
        destination[cuda.threadIdx.x] = helper(source, chained)

    assert _compile(kernel, types.int32[::1], types.int32[::1]).metadata[
        "ltoir"
    ]


def test_none_alias_reports_descriptor_error_before_type_inference():
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        empty = None
        storage = numba_coop.TempStorage()
        if source[0] > 0:
            storage = empty
        payload = numba_coop.ThreadData(1, types.int32)
        numba_coop.load(
            numba_coop.this_block(), source, payload, temp_storage=storage
        )
        destination[cuda.threadIdx.x] = payload[0]

    with pytest.raises(
        (GroupRewriteError, TypingError), match="TempStorage.*on every path"
    ):
        _compile(kernel, types.int32[::1], types.int32[::1])


def test_standalone_primitive_helper_reports_inline_requirement():
    @cuda.jit(device=True, inline="never")
    def standalone(destination, value):
        numba_coop.store(numba_coop.this_block(), destination, value)

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
def test_literal_unroll_cannot_determine_cooperative_payload_shape(
    payload_kind,
):
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        for count in literal_unroll((1, 2)):
            if payload_kind == "thread_data":
                payload = numba_coop.ThreadData(count, types.int32)
            elif payload_kind == "local_array":
                payload = cuda.local.array(count, types.int32)
            else:
                payload = cuda.local.array(shape=count, dtype=types.int32)
            numba_coop.load(numba_coop.this_block(), source, payload)
            destination[cuda.threadIdx.x] = payload[0]

    with pytest.raises(
        (GroupRewriteError, TypingError),
        match="does not support literal_unroll values",
    ):
        _compile(kernel, types.int32[::1], types.int32[::1])


def test_unrelated_literal_unroll_can_coexist_with_cooperative_calls():
    @cuda.jit(chip="sm_90")
    def kernel(destination):
        value = 0
        for item in literal_unroll((1, 2)):
            value += item
        numba_coop.store(numba_coop.this_block(), destination, value)

    assert _compile(kernel, types.int64[::1]).metadata["ltoir"]


def test_storage_free_operation_can_coexist_with_user_dynamic_shared_memory():
    @cuda.jit(chip="sm_90")
    def kernel(destination):
        mine = cuda.shared.array(0, types.int32)
        mine[cuda.threadIdx.x] = 42
        numba_coop.store(
            numba_coop.this_block(), destination, mine[cuda.threadIdx.x]
        )

    assert _compile(kernel, types.int32[::1]).metadata["ltoir"]


def test_literal_unroll_cannot_determine_cooperative_selector():
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        for algorithm in literal_unroll(("direct", "striped")):
            numba_coop.store(
                numba_coop.this_block(),
                destination,
                source[cuda.threadIdx.x],
                algorithm=algorithm,
            )

    with pytest.raises(
        (GroupRewriteError, TypingError),
        match="does not support literal_unroll values",
    ):
        _compile(kernel, types.int32[::1], types.int32[::1])


@pytest.mark.parametrize(
    "dynamic_backing", [False, True], ids=["static", "dynamic"]
)
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
        storage = numba_coop.TempStorage(size_in_bytes)
        payload = numba_coop.ThreadData(1, types.int32)
        payload[0] = mine[thread]
        numba_coop.store(
            numba_coop.this_block(),
            destination,
            payload,
            algorithm="transpose",
            temp_storage=storage,
        )

    if not dynamic_backing and user_shape == "static":
        assert _compile(kernel, types.int32[::1], types.int32[::1]).metadata[
            "ltoir"
        ]
    else:
        with pytest.raises(
            CoopSinglePhaseRewriteError,
            match=(
                r"shared-memory backing.*test_storage_diagnostics.py.*"
                r"would alias"
            ),
        ):
            _compile(kernel, types.int32[::1], types.int32[::1])


@pytest.mark.parametrize("argument", ["storage", "group_by"])
def test_literal_unroll_cannot_determine_cooperative_storage_or_partition(
    argument,
):
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        for count in literal_unroll((32, 64)):
            block = numba_coop.this_block()
            if argument == "storage":
                storage = numba_coop.TempStorage(count)
                payload = numba_coop.ThreadData(1, types.int32)
                payload[0] = source[cuda.threadIdx.x]
                numba_coop.store(
                    block,
                    destination,
                    payload,
                    algorithm="transpose",
                    temp_storage=storage,
                )
            else:
                group = block.group_by(count)
                numba_coop.store(group, destination, source[cuda.threadIdx.x])

    with pytest.raises(
        (GroupRewriteError, TypingError),
        match="does not support literal_unroll values",
    ):
        _compile(kernel, types.int32[::1], types.int32[::1])


def test_implicit_oversized_storage_rejects_user_static_shared_allocation(
    monkeypatch,
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

    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        mine = cuda.shared.array(1024, types.int32)
        thread = cuda.threadIdx.x
        mine[thread] = source[thread]
        payload = numba_coop.ThreadData(16, types.int32)
        for item in range(16):
            payload[item] = mine[thread]
        numba_coop.store(
            numba_coop.this_block(), destination, payload, algorithm="transpose"
        )

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            "dynamic shared-memory backing.*static cuda.shared.array.*"
            "would alias"
        ),
    ):
        _compile(kernel, types.int32[::1], types.int32[::1], block=(1024, 1, 1))


@pytest.mark.parametrize("kind", ["block", "mapped_warps", "cluster"])
@pytest.mark.parametrize("shape", [0, 128], ids=["dynamic", "static"])
def test_cudax_reduce_shared_memory_coexistence(kind, shape):
    @cuda.jit(device=True)
    def allocate():
        return cuda.shared.array(shape, types.int32)

    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        tile = allocate()
        thread = cuda.threadIdx.x
        tile[thread] = source[thread]
        cuda.syncthreads()
        if kind == "block":
            total = coop.sum(coop.this_block(), source[thread])
        elif kind == "mapped_warps":
            total = coop.sum(coop.this_block().group_by(2), source[thread])
        else:
            total = coop.sum(coop.this_cluster(), source[thread])
        destination[thread] = tile[thread] + total

    launch = {"block": (128, 1, 1)}
    if kind == "cluster":
        launch["cluster"] = (2, 1, 1)
    if shape:
        assert _compile(
            kernel, types.int32[::1], types.int32[::1], **launch
        ).metadata["ltoir"]
    else:
        with pytest.raises(
            CoopSinglePhaseRewriteError,
            match="CUDAX reduction.*static shared memory.*would alias",
        ):
            _compile(kernel, types.int32[::1], types.int32[::1], **launch)


@pytest.mark.parametrize("kind", ["block", "mapped_warps"])
def test_cudax_reduce_rejects_dynamic_cooperative_backing(monkeypatch, kind):
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
        scratch = coop.TempStorage(64 * 1024, auto_sync=True)
        items = coop.ThreadData(2, types.int32)
        coop.load(
            coop.this_block(),
            source,
            items,
            algorithm="transpose",
            temp_storage=scratch,
        )
        if source[0] >= 0:
            if kind == "block":
                total = coop.sum(coop.this_block(), items[0])
            else:
                total = coop.sum(coop.this_block().group_by(2), items[0])
            destination[cuda.threadIdx.x] = total

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="dynamic shared-memory backing.*CUDAX reduction.*would alias",
    ):
        _compile(kernel, types.int32[::1], types.int32[::1], block=(128, 1, 1))


@pytest.mark.parametrize("kind", ["warp", "logical_warp", "mapped_query"])
def test_shared_memory_free_group_providers_allow_user_dynamic_arrays(kind):
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        tile = cuda.shared.array(0, types.int32)
        thread = cuda.threadIdx.x
        tile[thread] = source[thread]
        if kind == "mapped_query":
            result = coop.this_block().group_by(2).rank()
        elif kind == "logical_warp":
            result = coop.sum(coop.this_warp().group_by(8), source[thread])
        else:
            result = coop.sum(coop.this_warp(), source[thread])
        destination[thread] = tile[thread] + result

    assert _compile(
        kernel, types.int32[::1], types.int32[::1], block=(128, 1, 1)
    ).metadata["ltoir"]
