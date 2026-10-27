# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check shared output shape, ownership, and per-thread sizing diagnostics."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")
import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

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
@pytest.mark.parametrize("count", [1, 4])
def test_shared_output_compiles(api, count):
    capacity = 32 * count

    @cuda.jit
    def kernel(source, result):
        shared = cuda.shared.array(capacity, source.dtype)
        status = api.load(
            api.this_block(), source, shared, items_per_thread=count
        )
        cuda.syncthreads()
        if status is None:
            result[cuda.threadIdx.x] = shared[cuda.threadIdx.x]

    _compile(kernel)


@pytest.mark.parametrize(
    "kind,message",
    [
        ("local", "shared array"),
        ("payload", "shared array"),
        ("global", "shared array"),
        ("capacity", "at least 128"),
        ("slice_capacity", "at least 128"),
        ("rank", "one-dimensional"),
        ("stride", "contiguous"),
        ("dtype", "dtype"),
        ("alignment", "alignment requires"),
        ("count", "positive compile-time integer"),
    ],
)
def test_invalid_shared_output(kind, message):
    @cuda.jit
    def kernel(source, result):
        if kind == "local":
            buffer = cuda.local.array(4, types.int32)
        elif kind == "payload":
            buffer = coop.ThreadData(items_per_thread=4)
        elif kind == "global":
            buffer = result
        elif kind == "capacity":
            buffer = cuda.shared.array(127, types.int32)
        elif kind == "rank":
            buffer = cuda.shared.array((32, 4), types.int32)
        elif kind == "dtype":
            buffer = cuda.shared.array(128, types.float32)
        else:
            buffer = cuda.shared.array(128, types.int32)
        if kind == "stride":
            output = buffer[::2]
        elif kind == "slice_capacity":
            output = buffer[1:]
        else:
            output = buffer
        if kind == "alignment":
            coop.load(
                coop.this_block(),
                source,
                output,
                items_per_thread=4,
                alignment=16,
            )
        elif kind == "count":
            coop.load(coop.this_block(), source, output, items_per_thread=0)
        else:
            coop.load(coop.this_block(), source, output, items_per_thread=4)

    with pytest.raises((TypeError, ValueError), match=message):
        _compile(kernel)
