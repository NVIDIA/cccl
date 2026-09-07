# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import pytest

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
torch = pytest.importorskip("torch")

import cuda.coop.cutlass as coop

from_dlpack = runtime.from_dlpack


@cute.kernel
def _group_load_store_kernel(
    values_in: cute.Tensor,
    block_out: cute.Tensor,
    reduced_out: cute.Tensor,
    partial_out: cute.Tensor,
    warp_out: cute.Tensor,
    scalar_out: cute.Tensor,
    array_out: cute.Tensor,
    rmem_out: cute.Tensor,
    tensorssa_out: cute.Tensor,
    valid_items: cutlass.Constexpr,
):
    block = coop.this_block()
    block_items = coop.ThreadData(2)
    coop.load(block, values_in, block_items, offset=3)
    coop.store(block, block_out, block_items, offset=5)
    reduced = coop.reduce(block, block_items)
    coop.store(block, reduced_out, reduced)

    partial_items = coop.ThreadData(2)
    coop.load(
        block,
        values_in,
        partial_items,
        valid_items=valid_items,
        oob_default=-777,
    )
    coop.store(block, partial_out, partial_items)

    warp = coop.this_warp()
    warp_items = coop.ThreadData(2)
    coop.load(warp, values_in, warp_items, algorithm="striped")
    coop.store(warp, warp_out, warp_items, algorithm="striped")

    tidx, _, _ = cute.arch.thread_idx()
    coop.store(block, scalar_out, values_in[tidx])

    values_array = cutlass.Array(values_in)
    output_array = cutlass.Array(array_out)
    array_items = coop.ThreadData(2)
    coop.load(block, values_array, array_items)
    coop.store(block, output_array, array_items)

    fragment_base = tidx * 2
    fragment = cute.make_rmem_tensor((1, 2), cutlass.Int32)
    fragment[0] = values_in[fragment_base]
    fragment[1] = values_in[fragment_base + 1]
    coop.store(block, rmem_out, fragment)
    coop.store(block, tensorssa_out, fragment.load())


@cute.jit
def _run_group_load_store(
    values_in: cute.Tensor,
    block_out: cute.Tensor,
    reduced_out: cute.Tensor,
    partial_out: cute.Tensor,
    warp_out: cute.Tensor,
    scalar_out: cute.Tensor,
    array_out: cute.Tensor,
    rmem_out: cute.Tensor,
    tensorssa_out: cute.Tensor,
    valid_items: cutlass.Constexpr,
):
    _group_load_store_kernel(
        values_in,
        block_out,
        reduced_out,
        partial_out,
        warp_out,
        scalar_out,
        array_out,
        rmem_out,
        tensorssa_out,
        valid_items,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_group_load_store_runtime_pipeline():
    cutlass.cuda.initialize_cuda_context()

    block_threads = 64
    items_per_thread = 2
    tile_items = block_threads * items_per_thread
    valid_items = tile_items - 7
    values_host = torch.arange(tile_items + 8, dtype=torch.int32)
    values_in = values_host.cuda()
    block_out = torch.full(
        (tile_items + 8,),
        -999,
        dtype=torch.int32,
        device="cuda",
    )
    reduced_out = torch.empty((block_threads,), dtype=torch.int32, device="cuda")
    partial_out = torch.empty((tile_items,), dtype=torch.int32, device="cuda")
    warp_out = torch.empty((tile_items,), dtype=torch.int32, device="cuda")
    scalar_out = torch.empty((block_threads,), dtype=torch.int32, device="cuda")
    array_out = torch.empty((tile_items,), dtype=torch.int32, device="cuda")
    rmem_out = torch.empty((tile_items,), dtype=torch.int32, device="cuda")
    tensorssa_out = torch.empty((tile_items,), dtype=torch.int32, device="cuda")

    _run_group_load_store(
        from_dlpack(values_in),
        from_dlpack(block_out),
        from_dlpack(reduced_out),
        from_dlpack(partial_out),
        from_dlpack(warp_out),
        from_dlpack(scalar_out),
        from_dlpack(array_out),
        from_dlpack(rmem_out),
        from_dlpack(tensorssa_out),
        valid_items,
    )
    torch.cuda.synchronize()

    expected_block = torch.full((tile_items + 8,), -999, dtype=torch.int32)
    expected_block[5 : 5 + tile_items] = values_host[3 : 3 + tile_items]
    expected_partial = torch.full((tile_items,), -777, dtype=torch.int32)
    expected_partial[:valid_items] = values_host[:valid_items]
    torch.testing.assert_close(block_out.cpu(), expected_block, atol=0, rtol=0)
    expected_reduced = torch.full(
        (block_threads,),
        int(values_host[3 : 3 + tile_items].sum()),
        dtype=torch.int32,
    )
    torch.testing.assert_close(reduced_out.cpu(), expected_reduced, atol=0, rtol=0)
    torch.testing.assert_close(partial_out.cpu(), expected_partial, atol=0, rtol=0)
    torch.testing.assert_close(warp_out.cpu(), values_host[:tile_items], atol=0, rtol=0)
    torch.testing.assert_close(
        array_out.cpu(), values_host[:tile_items], atol=0, rtol=0
    )
    torch.testing.assert_close(rmem_out.cpu(), values_host[:tile_items], atol=0, rtol=0)
    torch.testing.assert_close(
        tensorssa_out.cpu(), values_host[:tile_items], atol=0, rtol=0
    )
    torch.testing.assert_close(
        scalar_out.cpu(),
        values_host[:block_threads],
        atol=0,
        rtol=0,
    )
