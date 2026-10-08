# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile TopK kernels with real provider linkage and no visible GPU.

Pin device-target queries to SM90 while retaining the normal compiler and
linker. Runtime counts keep the generated count checks in the provider.
Explicit scratch checks the caller-owned storage ABI. The default
``auto_sync=False`` inserts no reuse barrier, so ``bar.sync`` comes from
CUB's internal TopK barriers.
"""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")
import numba_cuda_mlir.tools as compiler_tools
from numba_cuda_mlir import cuda, types

from cuda import coop
from cuda.coop.numba_mlir import _types

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.mark.parametrize("selection", ["min", "max"])
@pytest.mark.parametrize("pairs", [False, True])
@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_topk_production_compile_links_checked_provider(
    monkeypatch, selection, pairs, items_per_thread
):
    """Check linked TopK code for count traps and CUB block barriers.

    Compile all four selection forms with runtime counts and explicit scratch.
    Pairs use float64 keys and independent int64 values. Cubin and external
    link items confirm provider linkage; PTX checks confirm count validation
    and CUB's internal synchronization. The test does not launch the kernel or
    inspect its selected values.
    """

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    device = SimpleNamespace(compute_capability=(9, 0))
    monkeypatch.setattr(_types.cuda, "get_current_device", lambda: device)
    monkeypatch.setattr(
        compiler_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    topk = getattr(coop, f"topk_{selection}_{'pairs' if pairs else 'keys'}")

    @cuda.jit(chip="sm_90")
    def kernel(source, destination, k, count, items_per_thread):
        block = coop.this_block()
        scratch = coop.TempStorage()
        keys = coop.ThreadData(items_per_thread, dtype=types.float64)
        coop.load(block, source, keys)
        if pairs:
            values = coop.ThreadData(items_per_thread, dtype=types.int64)
            for item in range(items_per_thread):
                values[item] = types.int64(
                    cuda.threadIdx.x * items_per_thread + item
                )
            selected, indices = topk(
                block,
                keys,
                values,
                k=k,
                valid_items=count,
                temp_storage=scratch,
            )
            destination[cuda.threadIdx.x * items_per_thread] = (
                selected[0] + indices[0]
            )
        else:
            selected = topk(
                block, keys, k=k, valid_items=count, temp_storage=scratch
            )
            destination[cuda.threadIdx.x * items_per_thread] = selected[0]

    signature = types.void(
        types.float64[::1],
        types.float64[::1],
        types.int64,
        types.int64,
        types.IntegerLiteral(items_per_thread),
    )
    launch = (
        ("grid", (1, 1, 1)),
        ("block", (64, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    result = kernel._compile_launch_config_signature(signature, launch)
    assert result.metadata["cubin"]
    assert result.metadata["linked_external_link_items"]
    ptx = next(iter(kernel.inspect_lto_ptx().values()))
    assert "trap;" in ptx
    assert "bar.sync" in ptx
