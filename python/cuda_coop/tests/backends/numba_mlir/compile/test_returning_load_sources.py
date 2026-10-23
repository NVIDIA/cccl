# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check Load source inference and the qualified typed-output fallback."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop
from cuda.coop.numba_mlir._compiler._group_errors import (
    UnknownLoadStoreDtypeError,
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


def _compile(kernel):
    return kernel._compile_launch_config_signature(
        types.void(types.int32[::1], types.int32[::1]),
        (
            ("grid", (1, 1, 1)),
            ("block", (32, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize(
    "returning", [False, True], ids=["output", "returning"]
)
def test_load_infers_contiguous_source_slice_dtype(api, returning):
    @cuda.jit
    def kernel(source, destination):
        group = api.this_block()
        memory = source[1:]
        if returning:
            values = api.load(group, memory, items_per_thread=1)
        else:
            values = api.ThreadData(items_per_thread=1)
            api.load(group, memory, values)
        api.store(group, destination, values)

    _compile(kernel)


@pytest.mark.parametrize("sliced", [False, True], ids=["array", "slice"])
@pytest.mark.parametrize(
    "returning", [False, True], ids=["output", "returning"]
)
def test_qualified_shared_source_requires_typed_output(sliced, returning):
    @cuda.jit
    def kernel(source, destination):
        group = numba_coop.this_block()
        shared = cuda.shared.array(33, types.int32)
        thread = cuda.threadIdx.x
        shared[thread + 1] = source[thread]
        if thread == 0:
            shared[0] = source[0]
        cuda.syncthreads()
        if sliced:
            memory = shared[1:]
        else:
            memory = shared
        if returning:
            values = numba_coop.load(group, memory, items_per_thread=1)
        else:
            values = numba_coop.ThreadData(
                items_per_thread=1, dtype=types.int32
            )
            numba_coop.load(group, memory, values)
        numba_coop.store(group, destination, values)

    if returning:
        with pytest.raises(
            UnknownLoadStoreDtypeError, match="could not infer a dtype"
        ):
            _compile(kernel)
    else:
        _compile(kernel)
