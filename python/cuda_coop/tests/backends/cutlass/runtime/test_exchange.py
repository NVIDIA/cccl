# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.runtime import (
    EXCHANGE_TEMP_STORAGE as _EXCHANGE_TEMP_STORAGE,
)
from ..support.runtime import (
    Float32,
    Int32,
    coop,
    cute,
    cutlass,
    from_dlpack,
    runtime_pytestmark,
    torch,
)

pytestmark = runtime_pytestmark


@cute.kernel
def _warp_exchange_subwarp_kernel(
    values_in: cute.Tensor,
    ranks_in: cute.Tensor,
    striped_to_blocked_out: cute.Tensor,
    blocked_to_striped_out: cute.Tensor,
    scatter_to_striped_out: cute.Tensor,
):
    striped_items = coop._warp.load(
        values_in,
        items_per_thread=4,
        algorithm="striped",
        dtype=Int32,
        threads_in_warp=16,
    )
    blocked_items = coop._warp.exchange_striped_to_blocked(
        striped_items,
        threads_in_warp=16,
    )
    coop._warp.store(
        striped_to_blocked_out,
        blocked_items,
        threads_in_warp=16,
    )

    direct_items = coop._warp.load(
        values_in,
        items_per_thread=4,
        dtype=Int32,
        threads_in_warp=16,
    )
    striped_result = coop._warp.exchange_blocked_to_striped(
        direct_items,
        threads_in_warp=16,
    )
    coop._warp.store(
        blocked_to_striped_out,
        striped_result,
        algorithm="striped",
        threads_in_warp=16,
    )

    rank_items = coop._warp.load(
        ranks_in,
        items_per_thread=4,
        dtype=Int32,
        threads_in_warp=16,
    )
    scatter_result = coop._warp.exchange(
        direct_items,
        ranks=rank_items,
        warp_exchange_type=coop._warp.WarpExchangeType.ScatterToStriped,
        threads_in_warp=16,
    )
    coop._warp.store(
        scatter_to_striped_out,
        scatter_result,
        algorithm="striped",
        threads_in_warp=16,
    )


@cute.jit
def _run_warp_exchange_subwarp(
    values_in: cute.Tensor,
    ranks_in: cute.Tensor,
    striped_to_blocked_out: cute.Tensor,
    blocked_to_striped_out: cute.Tensor,
    scatter_to_striped_out: cute.Tensor,
):
    _warp_exchange_subwarp_kernel(
        values_in,
        ranks_in,
        striped_to_blocked_out,
        blocked_to_striped_out,
        scatter_to_striped_out,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1))


def test_provider_warp_exchange_runtime_subwarp_thread_data():
    cutlass.cuda.initialize_cuda_context()

    threads_in_warp = 16
    block_x = 32
    items_per_thread = 4
    tile_items = threads_in_warp * items_per_thread
    total_items = block_x * items_per_thread
    values_host = torch.arange(total_items, dtype=torch.int32) * 5 - 11
    ranks_host = torch.empty((total_items,), dtype=torch.int32)
    expected_scatter = torch.empty_like(values_host)
    for tile_base in range(0, total_items, tile_items):
        for local_idx in range(tile_items):
            rank = tile_items - 1 - local_idx
            ranks_host[tile_base + local_idx] = rank
            expected_scatter[tile_base + rank] = values_host[tile_base + local_idx]

    values_in = values_host.cuda()
    ranks_in = ranks_host.cuda()
    striped_to_blocked_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )
    blocked_to_striped_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )
    scatter_to_striped_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )

    _run_warp_exchange_subwarp(
        from_dlpack(values_in),
        from_dlpack(ranks_in),
        from_dlpack(striped_to_blocked_out),
        from_dlpack(blocked_to_striped_out),
        from_dlpack(scatter_to_striped_out),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        striped_to_blocked_out.cpu(), values_host, atol=0, rtol=0
    )
    torch.testing.assert_close(
        blocked_to_striped_out.cpu(), values_host, atol=0, rtol=0
    )
    torch.testing.assert_close(
        scatter_to_striped_out.cpu(), expected_scatter, atol=0, rtol=0
    )


@cute.kernel
def _warp_exchange_float32_kernel(
    values_in: cute.Tensor,
    ranks_in: cute.Tensor,
    striped_to_blocked_out: cute.Tensor,
    blocked_to_striped_out: cute.Tensor,
    scatter_to_striped_out: cute.Tensor,
):
    striped_items = coop._warp.load(
        values_in,
        items_per_thread=3,
        algorithm="striped",
        dtype=Float32,
        threads_in_warp=8,
    )
    blocked_items = coop._warp.exchange_striped_to_blocked(
        striped_items,
        threads_in_warp=8,
    )
    coop._warp.store(
        striped_to_blocked_out,
        blocked_items,
        threads_in_warp=8,
    )

    direct_items = coop._warp.load(
        values_in,
        items_per_thread=3,
        dtype=Float32,
        threads_in_warp=8,
    )
    striped_result = coop._warp.exchange_blocked_to_striped(
        direct_items,
        threads_in_warp=8,
    )
    coop._warp.store(
        blocked_to_striped_out,
        striped_result,
        algorithm="striped",
        threads_in_warp=8,
    )

    rank_items = coop._warp.load(
        ranks_in,
        items_per_thread=3,
        dtype=Int32,
        threads_in_warp=8,
    )
    scatter_result = coop._warp.exchange_scatter_to_striped(
        direct_items,
        rank_items,
        threads_in_warp=8,
    )
    coop._warp.store(
        scatter_to_striped_out,
        scatter_result,
        algorithm="striped",
        threads_in_warp=8,
    )


@cute.jit
def _run_warp_exchange_float32(
    values_in: cute.Tensor,
    ranks_in: cute.Tensor,
    striped_to_blocked_out: cute.Tensor,
    blocked_to_striped_out: cute.Tensor,
    scatter_to_striped_out: cute.Tensor,
):
    _warp_exchange_float32_kernel(
        values_in,
        ranks_in,
        striped_to_blocked_out,
        blocked_to_striped_out,
        scatter_to_striped_out,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1))


def test_provider_warp_exchange_runtime_float32_logical_warps():
    cutlass.cuda.initialize_cuda_context()

    threads_in_warp = 8
    block_x = 32
    items_per_thread = 3
    tile_items = threads_in_warp * items_per_thread
    total_items = block_x * items_per_thread
    values_host = torch.arange(total_items, dtype=torch.float32) * 0.5 - 7.25
    ranks_host = torch.empty((total_items,), dtype=torch.int32)
    expected_scatter = torch.empty_like(values_host)
    for tile_base in range(0, total_items, tile_items):
        for local_idx in range(tile_items):
            rank = tile_items - 1 - local_idx
            ranks_host[tile_base + local_idx] = rank
            expected_scatter[tile_base + rank] = values_host[tile_base + local_idx]

    values_in = values_host.cuda()
    ranks_in = ranks_host.cuda()
    striped_to_blocked_out = torch.zeros(
        (total_items,), dtype=torch.float32, device="cuda"
    )
    blocked_to_striped_out = torch.zeros(
        (total_items,), dtype=torch.float32, device="cuda"
    )
    scatter_to_striped_out = torch.zeros(
        (total_items,), dtype=torch.float32, device="cuda"
    )

    _run_warp_exchange_float32(
        from_dlpack(values_in),
        from_dlpack(ranks_in),
        from_dlpack(striped_to_blocked_out),
        from_dlpack(blocked_to_striped_out),
        from_dlpack(scatter_to_striped_out),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        striped_to_blocked_out.cpu(), values_host, atol=0, rtol=0
    )
    torch.testing.assert_close(
        blocked_to_striped_out.cpu(), values_host, atol=0, rtol=0
    )
    torch.testing.assert_close(
        scatter_to_striped_out.cpu(), expected_scatter, atol=0, rtol=0
    )


@cute.kernel
def _exchange_temp_kernel(
    values_in: cute.Tensor,
    ranks_in: cute.Tensor,
    reverse_ranks_in: cute.Tensor,
    guarded_ranks_in: cute.Tensor,
    valid_flags_in: cute.Tensor,
    striped_to_blocked_out: cute.Tensor,
    blocked_to_striped_out: cute.Tensor,
    blocked_to_warp_striped_out: cute.Tensor,
    warp_striped_to_blocked_out: cute.Tensor,
    scatter_to_blocked_out: cute.Tensor,
    scatter_to_striped_out: cute.Tensor,
    scatter_to_striped_guarded_out: cute.Tensor,
    scatter_to_striped_flagged_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_id = tidx // 32
    lane_id = tidx - warp_id * 32
    warp_base = warp_id * 32 * 5
    direct_base = tidx * 5

    striped_items = coop._block.load(
        values_in,
        items_per_thread=5,
        algorithm="striped",
        dtype=Int32,
    )
    blocked_items = coop._block.exchange_striped_to_blocked(
        striped_items,
        temp_storage=_EXCHANGE_TEMP_STORAGE,
    )
    coop._block.store(striped_to_blocked_out, blocked_items)

    striped_ranks = coop._block.load(
        ranks_in,
        items_per_thread=5,
        algorithm="striped",
        dtype=Int32,
    )
    scatter_blocked = coop._block.exchange_scatter_to_blocked(
        striped_items,
        striped_ranks,
        temp_storage=_EXCHANGE_TEMP_STORAGE,
    )
    coop._block.store(scatter_to_blocked_out, scatter_blocked)

    direct_items = coop._block.load(values_in, items_per_thread=5, dtype=Int32)
    striped_result = coop._block.exchange_blocked_to_striped(
        direct_items,
        temp_storage=_EXCHANGE_TEMP_STORAGE,
    )
    coop._block.store(blocked_to_striped_out, striped_result, algorithm="striped")

    direct_ranks = coop._block.load(ranks_in, items_per_thread=5, dtype=Int32)
    scatter_striped = coop._block.exchange_scatter_to_striped(
        direct_items,
        direct_ranks,
        temp_storage=_EXCHANGE_TEMP_STORAGE,
    )
    coop._block.store(scatter_to_striped_out, scatter_striped, algorithm="striped")

    guarded_ranks = coop._block.load(guarded_ranks_in, items_per_thread=5, dtype=Int32)
    scatter_guarded = coop._block.exchange_scatter_to_striped_guarded(
        direct_items,
        guarded_ranks,
        temp_storage=_EXCHANGE_TEMP_STORAGE,
    )
    coop._block.store(
        scatter_to_striped_guarded_out, scatter_guarded, algorithm="striped"
    )

    valid_flags = coop._block.load(valid_flags_in, items_per_thread=5, dtype=Int32)
    reverse_ranks = coop._block.load(reverse_ranks_in, items_per_thread=5, dtype=Int32)
    scatter_flagged = coop._block.exchange(
        direct_items,
        ranks=reverse_ranks,
        valid_flags=valid_flags,
        block_exchange_type=coop._block.BlockExchangeType.ScatterToStripedFlagged,
        temp_storage=_EXCHANGE_TEMP_STORAGE,
    )
    coop._block.store(
        scatter_to_striped_flagged_out, scatter_flagged, algorithm="striped"
    )

    warp_striped_result = coop._block.exchange_blocked_to_warp_striped(
        direct_items,
        temp_storage=_EXCHANGE_TEMP_STORAGE,
    )
    warp_striped_items = coop.ThreadData.from_values(
        values_in[warp_base + lane_id + 0 * 32],
        values_in[warp_base + lane_id + 1 * 32],
        values_in[warp_base + lane_id + 2 * 32],
        values_in[warp_base + lane_id + 3 * 32],
        values_in[warp_base + lane_id + 4 * 32],
        dtype=Int32,
    )
    blocked_result = coop._block.exchange_warp_striped_to_blocked(
        warp_striped_items,
        temp_storage=_EXCHANGE_TEMP_STORAGE,
    )

    blocked_to_warp_striped_out[warp_base + lane_id + 0 * 32] = warp_striped_result[0]
    blocked_to_warp_striped_out[warp_base + lane_id + 1 * 32] = warp_striped_result[1]
    blocked_to_warp_striped_out[warp_base + lane_id + 2 * 32] = warp_striped_result[2]
    blocked_to_warp_striped_out[warp_base + lane_id + 3 * 32] = warp_striped_result[3]
    blocked_to_warp_striped_out[warp_base + lane_id + 4 * 32] = warp_striped_result[4]
    warp_striped_to_blocked_out[direct_base + 0] = blocked_result[0]
    warp_striped_to_blocked_out[direct_base + 1] = blocked_result[1]
    warp_striped_to_blocked_out[direct_base + 2] = blocked_result[2]
    warp_striped_to_blocked_out[direct_base + 3] = blocked_result[3]
    warp_striped_to_blocked_out[direct_base + 4] = blocked_result[4]


@cute.jit
def _run_exchange_temp(
    values_in: cute.Tensor,
    ranks_in: cute.Tensor,
    reverse_ranks_in: cute.Tensor,
    guarded_ranks_in: cute.Tensor,
    valid_flags_in: cute.Tensor,
    striped_to_blocked_out: cute.Tensor,
    blocked_to_striped_out: cute.Tensor,
    blocked_to_warp_striped_out: cute.Tensor,
    warp_striped_to_blocked_out: cute.Tensor,
    scatter_to_blocked_out: cute.Tensor,
    scatter_to_striped_out: cute.Tensor,
    scatter_to_striped_guarded_out: cute.Tensor,
    scatter_to_striped_flagged_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _exchange_temp_kernel(
        values_in,
        ranks_in,
        reverse_ranks_in,
        guarded_ranks_in,
        valid_flags_in,
        striped_to_blocked_out,
        blocked_to_striped_out,
        blocked_to_warp_striped_out,
        warp_striped_to_blocked_out,
        scatter_to_blocked_out,
        scatter_to_striped_out,
        scatter_to_striped_guarded_out,
        scatter_to_striped_flagged_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


def test_provider_exchange_runtime_thread_data_above_warp_compatibility_limit():
    cutlass.cuda.initialize_cuda_context()
    _EXCHANGE_TEMP_STORAGE.reset_uses()

    block_x = 64
    items_per_thread = 5
    total_items = block_x * items_per_thread
    values_host = torch.arange(total_items, dtype=torch.int32)
    ranks_host = torch.arange(total_items, dtype=torch.int32)
    reverse_ranks_host = torch.arange(total_items - 1, -1, -1, dtype=torch.int32)
    guarded_ranks_host = reverse_ranks_host.clone()
    guarded_invalid_inputs = torch.arange(0, total_items, 17, dtype=torch.long)
    guarded_ranks_host[guarded_invalid_inputs] = -1
    valid_flags_host = torch.ones((total_items,), dtype=torch.int32)
    values_in = values_host.cuda()
    ranks_in = ranks_host.cuda()
    reverse_ranks_in = reverse_ranks_host.cuda()
    guarded_ranks_in = guarded_ranks_host.cuda()
    valid_flags_in = valid_flags_host.cuda()
    striped_to_blocked_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )
    blocked_to_striped_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )
    blocked_to_warp_striped_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )
    warp_striped_to_blocked_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )
    scatter_to_blocked_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )
    scatter_to_striped_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )
    scatter_to_striped_guarded_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )
    scatter_to_striped_flagged_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )

    _run_exchange_temp(
        from_dlpack(values_in),
        from_dlpack(ranks_in),
        from_dlpack(reverse_ranks_in),
        from_dlpack(guarded_ranks_in),
        from_dlpack(valid_flags_in),
        from_dlpack(striped_to_blocked_out),
        from_dlpack(blocked_to_striped_out),
        from_dlpack(blocked_to_warp_striped_out),
        from_dlpack(warp_striped_to_blocked_out),
        from_dlpack(scatter_to_blocked_out),
        from_dlpack(scatter_to_striped_out),
        from_dlpack(scatter_to_striped_guarded_out),
        from_dlpack(scatter_to_striped_flagged_out),
        block_x,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        striped_to_blocked_out.cpu(), values_host, atol=0, rtol=0
    )
    torch.testing.assert_close(
        blocked_to_striped_out.cpu(), values_host, atol=0, rtol=0
    )
    torch.testing.assert_close(
        blocked_to_warp_striped_out.cpu(), values_host, atol=0, rtol=0
    )
    torch.testing.assert_close(
        warp_striped_to_blocked_out.cpu(), values_host, atol=0, rtol=0
    )
    torch.testing.assert_close(
        scatter_to_blocked_out.cpu(), values_host, atol=0, rtol=0
    )
    torch.testing.assert_close(
        scatter_to_striped_out.cpu(), values_host, atol=0, rtol=0
    )
    reverse_expected = torch.flip(values_host, dims=(0,))
    guarded_defined = torch.ones((total_items,), dtype=torch.bool)
    guarded_defined[reverse_ranks_host[guarded_invalid_inputs].to(torch.long)] = False
    torch.testing.assert_close(
        scatter_to_striped_guarded_out.cpu()[guarded_defined],
        reverse_expected[guarded_defined],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        scatter_to_striped_flagged_out.cpu(), reverse_expected, atol=0, rtol=0
    )


@cute.kernel
def _exchange_no_temp_kernel(
    values_in: cute.Tensor,
    striped_to_blocked_out: cute.Tensor,
    blocked_to_striped_out: cute.Tensor,
):
    striped_items = coop._block.load(
        values_in,
        items_per_thread=4,
        algorithm="striped",
        dtype=Int32,
    )
    blocked_items = coop._block.exchange_striped_to_blocked(striped_items)
    coop._block.store(striped_to_blocked_out, blocked_items)

    direct_items = coop._block.load(values_in, items_per_thread=4, dtype=Int32)
    striped_result = coop._block.exchange_blocked_to_striped(direct_items)
    coop._block.store(blocked_to_striped_out, striped_result, algorithm="striped")


@cute.kernel
def _exchange_register_payload_no_temp_kernel(
    values_in: cute.Tensor,
    striped_to_blocked_out: cute.Tensor,
    blocked_to_striped_out: cute.Tensor,
):
    striped_items = coop._block.load(
        values_in,
        items_per_thread=4,
        algorithm="striped",
        dtype=Int32,
    )
    striped_fragment = cute.make_rmem_tensor((1, 4), Int32)
    striped_fragment[0] = striped_items[0]
    striped_fragment[1] = striped_items[1]
    striped_fragment[2] = striped_items[2]
    striped_fragment[3] = striped_items[3]
    blocked_items = coop._block.exchange_striped_to_blocked(striped_fragment)
    coop._block.store(striped_to_blocked_out, blocked_items)

    direct_items = coop._block.load(values_in, items_per_thread=4, dtype=Int32)
    direct_fragment = cute.make_rmem_tensor((1, 4), Int32)
    direct_fragment[0] = direct_items[0]
    direct_fragment[1] = direct_items[1]
    direct_fragment[2] = direct_items[2]
    direct_fragment[3] = direct_items[3]
    striped_result = coop._block.exchange_blocked_to_striped(direct_fragment.load())
    coop._block.store(blocked_to_striped_out, striped_result, algorithm="striped")


@cute.jit
def _run_exchange_no_temp(
    values_in: cute.Tensor,
    striped_to_blocked_out: cute.Tensor,
    blocked_to_striped_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _exchange_no_temp_kernel(
        values_in,
        striped_to_blocked_out,
        blocked_to_striped_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_exchange_register_payload_no_temp(
    values_in: cute.Tensor,
    striped_to_blocked_out: cute.Tensor,
    blocked_to_striped_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _exchange_register_payload_no_temp_kernel(
        values_in,
        striped_to_blocked_out,
        blocked_to_striped_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@pytest.mark.parametrize("block_x", [16, 32])
@pytest.mark.parametrize("payload_kind", ("thread_data", "register_payload"))
def test_provider_exchange_runtime_multi_item_payloads_without_temp_storage(
    block_x: int,
    payload_kind: str,
):
    cutlass.cuda.initialize_cuda_context()

    items_per_thread = 4
    total_items = block_x * items_per_thread
    values_host = torch.arange(total_items, dtype=torch.int32)
    values_in = values_host.cuda()
    striped_to_blocked_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )
    blocked_to_striped_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )

    runner = (
        _run_exchange_register_payload_no_temp
        if payload_kind == "register_payload"
        else _run_exchange_no_temp
    )
    runner(
        from_dlpack(values_in),
        from_dlpack(striped_to_blocked_out),
        from_dlpack(blocked_to_striped_out),
        block_x,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        striped_to_blocked_out.cpu(), values_host, atol=0, rtol=0
    )
    torch.testing.assert_close(
        blocked_to_striped_out.cpu(), values_host, atol=0, rtol=0
    )


@cute.kernel
def _exchange_warp_no_temp_kernel(
    values_in: cute.Tensor,
    blocked_to_warp_striped_out: cute.Tensor,
    warp_striped_to_blocked_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    block_x, _, _ = cute.arch.block_dim()
    lane_id = tidx & 31
    direct_base = tidx * 4

    direct_items = coop._block.load(values_in, items_per_thread=4, dtype=Int32)
    warp_striped_result = coop._block.exchange_blocked_to_warp_striped(direct_items)
    warp_striped_items = coop.ThreadData.from_values(
        values_in[lane_id + 0 * block_x],
        values_in[lane_id + 1 * block_x],
        values_in[lane_id + 2 * block_x],
        values_in[lane_id + 3 * block_x],
        dtype=Int32,
    )
    blocked_result = coop._block.exchange_warp_striped_to_blocked(warp_striped_items)

    blocked_to_warp_striped_out[lane_id + 0 * block_x] = warp_striped_result[0]
    blocked_to_warp_striped_out[lane_id + 1 * block_x] = warp_striped_result[1]
    blocked_to_warp_striped_out[lane_id + 2 * block_x] = warp_striped_result[2]
    blocked_to_warp_striped_out[lane_id + 3 * block_x] = warp_striped_result[3]
    warp_striped_to_blocked_out[direct_base + 0] = blocked_result[0]
    warp_striped_to_blocked_out[direct_base + 1] = blocked_result[1]
    warp_striped_to_blocked_out[direct_base + 2] = blocked_result[2]
    warp_striped_to_blocked_out[direct_base + 3] = blocked_result[3]


@cute.jit
def _run_exchange_warp_no_temp(
    values_in: cute.Tensor,
    blocked_to_warp_striped_out: cute.Tensor,
    warp_striped_to_blocked_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _exchange_warp_no_temp_kernel(
        values_in,
        blocked_to_warp_striped_out,
        warp_striped_to_blocked_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@pytest.mark.parametrize("block_x", [16, 32])
def test_provider_exchange_runtime_warp_striped_without_temp_storage(block_x: int):
    cutlass.cuda.initialize_cuda_context()

    items_per_thread = 4
    total_items = block_x * items_per_thread
    values_host = torch.arange(total_items, dtype=torch.int32)
    values_in = values_host.cuda()
    blocked_to_warp_striped_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )
    warp_striped_to_blocked_out = torch.zeros(
        (total_items,), dtype=torch.int32, device="cuda"
    )

    _run_exchange_warp_no_temp(
        from_dlpack(values_in),
        from_dlpack(blocked_to_warp_striped_out),
        from_dlpack(warp_striped_to_blocked_out),
        block_x,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        blocked_to_warp_striped_out.cpu(), values_host, atol=0, rtol=0
    )
    torch.testing.assert_close(
        warp_striped_to_blocked_out.cpu(), values_host, atol=0, rtol=0
    )
