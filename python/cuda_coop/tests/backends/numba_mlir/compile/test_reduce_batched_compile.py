# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""GPU-free final linking and rejection checks for batched warp reductions."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")
import numba_cuda_mlir.tools as compiler_tools
from numba_cuda_mlir import cuda, types

from cuda import coop
from cuda.coop.numba_mlir import _types

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.fixture
def compile_kernel(monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    device = SimpleNamespace(compute_capability=(9, 0))
    monkeypatch.setattr(_types.cuda, "get_current_device", lambda: device)
    monkeypatch.setattr(
        compiler_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )

    def compile_(kernel, block=64):
        signature = types.void(types.float64[::1], types.float64[::1])
        launch = (
            ("grid", (1, 1, 1)),
            ("block", (block, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        )
        return kernel._compile_launch_config_signature(signature, launch)

    return compile_


@pytest.mark.parametrize("layout", ["striped", "blocked"])
def test_reduce_batched_links_native_provider(compile_kernel, layout):
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        block = coop.this_block()
        values = coop.ThreadData(33)
        coop.load(block, source, values)
        result = coop.reduce_batched(coop.this_warp(), values, output_layout=layout)
        destination[cuda.threadIdx.x * 2] = result[0]
        destination[cuda.threadIdx.x * 2 + 1] = result[1]

    compiled = compile_kernel(kernel)
    assert compiled.metadata["cubin"]
    assert compiled.metadata["linked_external_link_items"]


@pytest.mark.parametrize(
    ("kind", "layout", "message"),
    [("block", "striped", "group kind.*block"), ("warp", "broadcast", "output_layout")],
)
def test_reduce_batched_rejects_invalid_contract(compile_kernel, kind, layout, message):
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        values = coop.ThreadData(3)
        coop.load(coop.this_block(), source, values)
        if kind == "block":
            group = coop.this_block()
        else:
            group = coop.this_warp()
        result = coop.reduce_batched(group, values, output_layout=layout)
        destination[cuda.threadIdx.x] = result[0]

    with pytest.raises(Exception, match=message):
        compile_kernel(kernel)


def test_reduce_batched_rejects_partial_physical_warp(compile_kernel):
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        values = coop.ThreadData(3)
        coop.load(coop.this_block(), source, values)
        result = coop.reduce_batched(coop.this_warp(), values)
        destination[cuda.threadIdx.x] = result[0]

    with pytest.raises(Exception, match="warp|multiple of 32"):
        compile_kernel(kernel, block=48)


def test_reduce_batched_provider_dtype_bundle(compile_kernel):
    from cuda.coop.numba_mlir._compiler import _nvrtc
    from cuda.coop.numba_mlir._lowering._reduce_batched import reduce_batched

    context = _nvrtc.resolve_compile_context()
    collected = []
    for dtype_name in (
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
    ):
        for width in (8, 32):
            for layout in ("blocked", "striped"):
                with _types.collect_specializations() as items:
                    reduce_batched(
                        getattr(types, dtype_name),
                        64,
                        width,
                        33,
                        output_layout=layout,
                    )
                assert len(items) == 1
                items[0][0]._compile_context = context
                collected.extend(items)
    ltoir = _types.prepare_ltoir_bundle(
        [item[0] for item in collected],
        bundle_name="reduce_batched_all_numeric_types",
        allow_single=True,
        threads_by_algo={id(item[0]): item[1] for item in collected},
        block_threads_by_algo={id(item[0]): item[2] for item in collected},
    )
    assert isinstance(ltoir, bytes) and ltoir
