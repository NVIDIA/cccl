# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Production compile coverage for Numba-CUDA-MLIR group hierarchy methods."""

from types import SimpleNamespace

import pytest
from numba_cuda_mlir import types

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]

_BLOCK_THREADS = 64
_FIXED_COMPUTE_CAPABILITY = (9, 0)


def _production_compile_environment(monkeypatch: pytest.MonkeyPatch):
    import numba_cuda_mlir.tools as numba_mlir_tools
    from numba_cuda_mlir import cuda as compiler_cuda

    fixed_device = SimpleNamespace(compute_capability=_FIXED_COMPUTE_CAPABILITY)

    def fixed_compute_capability(as_type=str):
        assert as_type in (str, tuple)
        return _FIXED_COMPUTE_CAPABILITY if as_type is tuple else "sm_90"

    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        fixed_compute_capability,
    )
    monkeypatch.setattr(compiler_cuda, "get_current_device", lambda: fixed_device)
    return compiler_cuda


def test_production_kernel_compiles_physical_and_mapped_group_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = _production_compile_environment(monkeypatch)

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as portable_coop

    @cuda.jit(chip="sm_90")
    def kernel(output):
        thread_index = cuda.threadIdx.x
        thread = portable_coop.this_thread()
        warp = qualified_coop.this_warp()
        block = portable_coop.this_block()
        grid = qualified_coop.this_grid()
        lanes = qualified_coop.this_warp().group_by(8)
        warps = portable_coop.this_block().group_by(2)

        thread.sync()
        warp.sync_aligned()
        lanes.sync()
        lanes.sync_aligned()
        block.sync()

        output[0 * _BLOCK_THREADS + thread_index] = thread.rank("block")
        output[1 * _BLOCK_THREADS + thread_index] = warp.count("block")
        output[2 * _BLOCK_THREADS + thread_index] = block.rank("thread")
        output[3 * _BLOCK_THREADS + thread_index] = block.count("grid")
        output[4 * _BLOCK_THREADS + thread_index] = lanes.rank("thread")
        output[5 * _BLOCK_THREADS + thread_index] = lanes.count("warp")
        output[6 * _BLOCK_THREADS + thread_index] = warps.rank("warp")
        output[7 * _BLOCK_THREADS + thread_index] = warps.count("thread")
        output[8 * _BLOCK_THREADS + thread_index] = warps.is_member()
        output[9 * _BLOCK_THREADS + thread_index] = block.rank_as(
            types.uint64,
            "thread",
        )
        output[10 * _BLOCK_THREADS + thread_index] = thread.count_as(types.int16)
        output[11 * _BLOCK_THREADS + thread_index] = grid.rank("thread")
        output[12 * _BLOCK_THREADS + thread_index] = grid.count("block")
        output[13 * _BLOCK_THREADS + thread_index] = grid.is_member()
        output[14 * _BLOCK_THREADS + thread_index] = block.rank() + block.rank_as(
            types.uint32
        )
        output[15 * _BLOCK_THREADS + thread_index] = block.rank_as(int) + block.rank_as(
            types.int32
        )

    signature = types.void(types.uint64[::1])
    launch_config_key = (
        ("grid", (2, 1, 1)),
        ("block", (_BLOCK_THREADS, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    result = kernel._compile_launch_config_signature(
        signature,
        launch_config_key,
    )

    assert result.metadata["ltoir"]
    assert result.metadata["cubin"]
    assert result.metadata["linked_external_link_items"]

    ptx = next(iter(kernel.inspect_lto_ptx().values()))
    assert ".visible .entry" in ptx


def test_production_kernel_compiles_cluster_queries_and_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = _production_compile_environment(monkeypatch)

    import cuda.coop.numba_mlir as coop

    @cuda.jit(chip="sm_90")
    def kernel(output):
        cluster = coop.this_cluster()
        cluster.sync()
        output[cuda.threadIdx.x] = cluster.rank("block")
        output[_BLOCK_THREADS + cuda.threadIdx.x] = cluster.count("grid")
        output[2 * _BLOCK_THREADS + cuda.threadIdx.x] = cluster.is_member()
        cluster.sync_aligned()

    signature = types.void(types.uint64[::1])
    launch_config_key = (
        ("grid", (2, 1, 1)),
        ("block", (_BLOCK_THREADS, 1, 1)),
        ("sharedmem", 0),
        ("cluster", (2, 1, 1)),
    )
    result = kernel._compile_launch_config_signature(
        signature,
        launch_config_key,
    )

    assert result.metadata["ltoir"]
    assert result.metadata["cubin"]
    assert result.metadata["linked_external_link_items"]


def test_block_warp_queries_compile_with_a_partial_final_warp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = _production_compile_environment(monkeypatch)

    from cuda import coop

    @cuda.jit(chip="sm_90")
    def kernel(output):
        block = coop.this_block()
        output[cuda.threadIdx.x] = block.rank("warp")
        output[48 + cuda.threadIdx.x] = block.count("warp")

    signature = types.void(types.uint32[::1])
    launch_config_key = (
        ("grid", (1, 1, 1)),
        ("block", (48, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    result = kernel._compile_launch_config_signature(
        signature,
        launch_config_key,
    )

    assert result.metadata["ltoir"]
    assert result.metadata["cubin"]
    assert result.metadata["linked_external_link_items"]


def test_thread_parent_warp_queries_compile_with_a_subwarp_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = _production_compile_environment(monkeypatch)

    from cuda import coop

    @cuda.jit(chip="sm_90")
    def kernel(output):
        thread = coop.this_thread()
        output[cuda.threadIdx.x] = thread.rank("warp")
        output[16 + cuda.threadIdx.x] = thread.count("warp")

    signature = types.void(types.uint32[::1])
    launch_config_key = (
        ("grid", (1, 1, 1)),
        ("block", (16, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    result = kernel._compile_launch_config_signature(
        signature,
        launch_config_key,
    )

    assert result.metadata["ltoir"]
    assert result.metadata["cubin"]
    assert result.metadata["linked_external_link_items"]
