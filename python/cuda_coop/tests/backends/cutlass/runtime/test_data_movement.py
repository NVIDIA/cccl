# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os

import pytest

from ....support.paths import REPO_ROOT

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("requires a CUDA-capable PyTorch runtime", allow_module_level=True)

coop = pytest.importorskip("cuda.coop.cutlass")

from_dlpack = runtime.from_dlpack

pytestmark = [
    pytest.mark.backend_cutlass,
    pytest.mark.runtime,
    pytest.mark.gpu,
]

_BLOCK_THREADS = 64
_WARP_THREADS = 32
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
_OUTPUT_SEGMENTS = 4
_PARTIAL_BLOCK_ITEMS = 95
_PARTIAL_WARP_ITEMS = 41


@pytest.fixture(scope="module", autouse=True)
def _isolated_provider_cache(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("cuda-coop-cutlass-movement-runtime")
    env_values = {
        "CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT": "ltoir",
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR": os.fspath(cache_dir),
    }
    if os.environ.get("CUDA_COOP_CUTLASS_FINAL_LINK_TEST") != "1":
        env_values["CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT"] = os.fspath(REPO_ROOT)
    original = {name: os.environ.get(name) for name in env_values}
    os.environ.update(env_values)
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@cute.kernel
def _data_movement_kernel(values: cute.Tensor, output: cute.Tensor):
    block = coop.this_block()
    block_items = coop.ThreadData(_ITEMS_PER_THREAD)
    loaded_block = coop.load(block, values, block_items)
    coop.store(block, output, loaded_block)

    warp = coop.this_warp()
    warp_items = coop.ThreadData(_ITEMS_PER_THREAD)
    loaded_warp = coop.load(warp, values, warp_items)
    coop.store(warp, output, loaded_warp, offset=_TILE_ITEMS)

    partial_block_items = coop.ThreadData(_ITEMS_PER_THREAD)
    partial_block = coop.load(
        block,
        values,
        partial_block_items,
        valid_items=_PARTIAL_BLOCK_ITEMS,
    )
    coop.store(block, output, partial_block, offset=2 * _TILE_ITEMS)

    partial_warp_items = coop.ThreadData(_ITEMS_PER_THREAD)
    partial_warp = coop.load(
        warp,
        values,
        partial_warp_items,
        valid_items=_PARTIAL_WARP_ITEMS,
    )
    coop.store(warp, output, partial_warp, offset=3 * _TILE_ITEMS)


@cute.jit
def _run_data_movement(values: cute.Tensor, output: cute.Tensor):
    _data_movement_kernel(values, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


@cute.kernel
def _fixed_storage_kernel(values: cute.Tensor, output: cute.Tensor):
    block = coop.this_block()
    storage = coop.TempStorage(4096, alignment=16)
    items = coop.ThreadData(_ITEMS_PER_THREAD)
    loaded = coop.load(block, values, items, temp_storage=storage)
    coop.store(block, output, loaded, temp_storage=storage)


@cute.jit
def _run_fixed_storage(values: cute.Tensor, output: cute.Tensor):
    _fixed_storage_kernel(values, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


def test_block_and_warp_data_movement_match_independent_oracles() -> None:
    cutlass.cuda.initialize_cuda_context()
    values_host = torch.arange(_TILE_ITEMS, dtype=torch.int32)
    values = values_host.cuda()
    output = torch.full(
        (_OUTPUT_SEGMENTS * _TILE_ITEMS,),
        -1,
        dtype=torch.int32,
        device="cuda",
    )

    _run_data_movement(from_dlpack(values), from_dlpack(output))
    torch.cuda.synchronize()
    segments = output.cpu().reshape(_OUTPUT_SEGMENTS, _TILE_ITEMS)

    torch.testing.assert_close(segments[0], values_host, atol=0, rtol=0)
    torch.testing.assert_close(segments[1], values_host, atol=0, rtol=0)
    torch.testing.assert_close(
        segments[2, :_PARTIAL_BLOCK_ITEMS],
        values_host[:_PARTIAL_BLOCK_ITEMS],
        atol=0,
        rtol=0,
    )
    for begin in range(0, _TILE_ITEMS, _WARP_THREADS * _ITEMS_PER_THREAD):
        torch.testing.assert_close(
            segments[
                3,
                begin : begin + _PARTIAL_WARP_ITEMS,
            ],
            values_host[begin : begin + _PARTIAL_WARP_ITEMS],
            atol=0,
            rtol=0,
        )


def test_fixed_capacity_storage_reaches_block_load_and_store() -> None:
    cutlass.cuda.initialize_cuda_context()
    values_host = torch.arange(_TILE_ITEMS, dtype=torch.int32)
    values = values_host.cuda()
    output = torch.full_like(values, -1)

    _run_fixed_storage(from_dlpack(values), from_dlpack(output))
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), values_host, atol=0, rtol=0)
