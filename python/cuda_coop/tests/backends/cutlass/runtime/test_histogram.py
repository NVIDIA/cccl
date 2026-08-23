# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.runtime import (
    HISTOGRAM_TEMP_STORAGE as _HISTOGRAM_TEMP_STORAGE,
)
from ..support.runtime import (
    LAUNCH_CASES as _LAUNCH_CASES,
)
from ..support.runtime import (
    Int64,
    Uint8,
    Uint32,
    coop,
    cute,
    cutlass,
    from_dlpack,
    runtime_pytestmark,
    torch,
)

pytestmark = runtime_pytestmark


@cute.kernel
def _histogram_kernel(
    values_in: cute.Tensor,
    hist_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    items_per_thread = cutlass.const_expr(2)
    bins = cutlass.const_expr(32)
    bins_per_thread = cutlass.const_expr(2)
    base = tidx * items_per_thread
    samples = cute.make_rmem_tensor((1, 2), Uint8)
    samples[0] = values_in[base + 0]
    samples[1] = values_in[base + 1]
    counts = coop._block.histogram(
        samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=Int64,
    )
    if tidx < block_x:
        hist_out[tidx] = counts[0]
        hist_out[tidx + block_x] = counts[1]


@cute.kernel
def _histogram_temp_kernel(
    values_in: cute.Tensor,
    hist_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    items_per_thread = cutlass.const_expr(2)
    bins = cutlass.const_expr(32)
    bins_per_thread = cutlass.const_expr(2)
    base = tidx * items_per_thread
    samples = cute.make_rmem_tensor((1, 2), Uint8)
    samples[0] = values_in[base + 0]
    samples[1] = values_in[base + 1]
    counts = coop._block.histogram(
        samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=Int64,
        temp_storage=_HISTOGRAM_TEMP_STORAGE,
    )
    if tidx < block_x:
        hist_out[tidx] = counts[0]
        hist_out[tidx + block_x] = counts[1]


@cute.jit
def _run_histogram(
    values_in: cute.Tensor,
    hist_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _histogram_kernel(values_in, hist_out, block_x).launch(
        grid=(1, 1, 1), block=(block_x, 1, 1)
    )


@cute.jit
def _run_histogram_temp(
    values_in: cute.Tensor,
    hist_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _histogram_temp_kernel(values_in, hist_out, block_x).launch(
        grid=(1, 1, 1), block=(block_x, 1, 1)
    )


@cute.kernel
def _histogram_root_scoped_parity_kernel(
    values_in: cute.Tensor,
    root_out: cute.Tensor,
    scoped_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    items_per_thread = cutlass.const_expr(2)
    bins = cutlass.const_expr(32)
    bins_per_thread = cutlass.const_expr(2)
    base = tidx * items_per_thread
    samples = cute.make_rmem_tensor((1, 2), Uint8)
    samples[0] = values_in[base + 0]
    samples[1] = values_in[base + 1]
    root_counts = coop.histogram(
        coop.this_block(),
        samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=Int64,
    )
    scoped_counts = coop._block.histogram(
        samples.load(),
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=Int64,
    )
    root_out[tidx] = root_counts[0]
    root_out[tidx + 64] = root_counts[1]
    scoped_out[tidx] = scoped_counts[0]
    scoped_out[tidx + 64] = scoped_counts[1]


@cute.jit
def _run_histogram_root_scoped_parity(
    values_in: cute.Tensor,
    root_out: cute.Tensor,
    scoped_out: cute.Tensor,
):
    _histogram_root_scoped_parity_kernel(
        values_in,
        root_out,
        scoped_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_histogram_runtime(block_x: int, use_temp_storage: bool):
    cutlass.cuda.initialize_cuda_context()
    _HISTOGRAM_TEMP_STORAGE.reset_uses()

    items_per_thread = 2
    bins = 32
    bins_per_thread = 2
    total_items = block_x * items_per_thread
    values_host = torch.tensor(
        [((idx * 7 + idx // 3) % bins) for idx in range(total_items)],
        dtype=torch.uint8,
    )
    values_in = values_host.cuda()
    hist_out = torch.zeros(
        (block_x * bins_per_thread,),
        dtype=torch.int64,
        device="cuda",
    )

    if use_temp_storage:
        _run_histogram_temp(
            from_dlpack(values_in),
            from_dlpack(hist_out),
            block_x,
        )
    else:
        _run_histogram(
            from_dlpack(values_in),
            from_dlpack(hist_out),
            block_x,
        )
    torch.cuda.synchronize()

    counts = torch.bincount(values_host.to(torch.int64), minlength=bins).to(torch.int64)
    expected = torch.zeros((block_x * bins_per_thread,), dtype=torch.int64)
    for tidx in range(block_x):
        for item_idx in range(bins_per_thread):
            bin_idx = tidx + item_idx * block_x
            if bin_idx < bins:
                expected[tidx + item_idx * block_x] = counts[bin_idx]
    torch.testing.assert_close(hist_out.cpu(), expected, atol=0, rtol=0)


def test_provider_histogram_root_scoped_runtime_parity():
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.tensor(
        [((idx * 7 + idx // 3) % 32) for idx in range(128)],
        dtype=torch.uint8,
    )
    values_in = values_host.cuda()
    root_out = torch.zeros((128,), dtype=torch.int64, device="cuda")
    scoped_out = torch.zeros((128,), dtype=torch.int64, device="cuda")

    _run_histogram_root_scoped_parity(
        from_dlpack(values_in),
        from_dlpack(root_out),
        from_dlpack(scoped_out),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(root_out, scoped_out, atol=0, rtol=0)
    expected = torch.bincount(values_host.to(torch.int64), minlength=32)
    torch.testing.assert_close(root_out[:32].cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(
        root_out[32:].cpu(),
        torch.zeros(96, dtype=torch.int64),
        atol=0,
        rtol=0,
    )


@cute.kernel
def _histogram_u8_u32_256_kernel(
    values_in: cute.Tensor,
    hist_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    bins_per_thread: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    items_per_thread = cutlass.const_expr(4)
    bins = cutlass.const_expr(256)
    base = tidx * items_per_thread
    samples = coop.ThreadData.from_values(
        values_in[base + 0],
        values_in[base + 1],
        values_in[base + 2],
        values_in[base + 3],
        dtype=Uint8,
    )
    counts = coop._block.histogram(
        samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=Uint32,
        algorithm="sort",
    )
    out_base = tidx * bins_per_thread
    hist_out[out_base + 0] = counts[0]
    hist_out[out_base + 1] = counts[1]
    hist_out[out_base + 2] = counts[2]
    hist_out[out_base + 3] = counts[3]
    if bins_per_thread > 4:
        hist_out[out_base + 4] = counts[4]
        hist_out[out_base + 5] = counts[5]
        hist_out[out_base + 6] = counts[6]
        hist_out[out_base + 7] = counts[7]


@cute.kernel
def _histogram_u8_u32_256_temp_kernel(
    values_in: cute.Tensor,
    hist_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    bins_per_thread: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    items_per_thread = cutlass.const_expr(4)
    bins = cutlass.const_expr(256)
    base = tidx * items_per_thread
    samples = coop.ThreadData.from_values(
        values_in[base + 0],
        values_in[base + 1],
        values_in[base + 2],
        values_in[base + 3],
        dtype=Uint8,
    )
    counts = coop._block.histogram(
        samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=Uint32,
        algorithm="sort",
        temp_storage=_HISTOGRAM_TEMP_STORAGE,
    )
    out_base = tidx * bins_per_thread
    hist_out[out_base + 0] = counts[0]
    hist_out[out_base + 1] = counts[1]
    hist_out[out_base + 2] = counts[2]
    hist_out[out_base + 3] = counts[3]
    if bins_per_thread > 4:
        hist_out[out_base + 4] = counts[4]
        hist_out[out_base + 5] = counts[5]
        hist_out[out_base + 6] = counts[6]
        hist_out[out_base + 7] = counts[7]


@cute.jit
def _run_histogram_u8_u32_256(
    values_in: cute.Tensor,
    hist_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    bins_per_thread: cutlass.Constexpr,
):
    _histogram_u8_u32_256_kernel(
        values_in,
        hist_out,
        block_x,
        bins_per_thread,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_histogram_u8_u32_256_temp(
    values_in: cute.Tensor,
    hist_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    bins_per_thread: cutlass.Constexpr,
):
    _histogram_u8_u32_256_temp_kernel(
        values_in,
        hist_out,
        block_x,
        bins_per_thread,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    [(32, False), (32, True)],
)
def test_provider_histogram_runtime_u8_u32_256_bins(
    block_x: int,
    use_temp_storage: bool,
):
    cutlass.cuda.initialize_cuda_context()
    _HISTOGRAM_TEMP_STORAGE.reset_uses()

    items_per_thread = 4
    bins = 256
    bins_per_thread = (bins + block_x - 1) // block_x
    total_items = block_x * items_per_thread
    values_host = torch.tensor(
        [((idx * 31 + idx // 5) % bins) for idx in range(total_items)],
        dtype=torch.uint8,
    )
    values_in = values_host.cuda()
    hist_out = torch.zeros(
        (block_x * bins_per_thread,),
        dtype=torch.uint32,
        device="cuda",
    )

    if use_temp_storage:
        _run_histogram_u8_u32_256_temp(
            from_dlpack(values_in),
            from_dlpack(hist_out),
            block_x,
            bins_per_thread,
        )
    else:
        _run_histogram_u8_u32_256(
            from_dlpack(values_in),
            from_dlpack(hist_out),
            block_x,
            bins_per_thread,
        )
    torch.cuda.synchronize()

    counts = torch.bincount(values_host.to(torch.int64), minlength=bins).to(
        torch.uint32
    )
    expected = torch.zeros((block_x * bins_per_thread,), dtype=torch.uint32)
    for tidx in range(block_x):
        for item_idx in range(bins_per_thread):
            bin_idx = tidx + item_idx * block_x
            if bin_idx < bins:
                expected[tidx * bins_per_thread + item_idx] = counts[bin_idx]
    torch.testing.assert_close(hist_out.cpu(), expected, atol=0, rtol=0)
