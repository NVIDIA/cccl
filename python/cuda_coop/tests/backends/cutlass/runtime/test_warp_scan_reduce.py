# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from ..support.runtime import (
    FLOAT32_LOWEST as _FLOAT32_LOWEST,
)
from ..support.runtime import (
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
def _warp_scan_reduce_float32_kernel(
    values_in: cute.Tensor,
    exclusive_sum_out: cute.Tensor,
    inclusive_sum_out: cute.Tensor,
    exclusive_max_out: cute.Tensor,
    inclusive_max_out: cute.Tensor,
    sum_out: cute.Tensor,
    max_out: cute.Tensor,
    min_out: cute.Tensor,
    reduce_max_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    logical_warp_id = tidx // cutlass.Int32(16)
    lane_id = tidx - logical_warp_id * cutlass.Int32(16)
    value = values_in[tidx]

    exclusive_sum_out[tidx] = coop._warp.exclusive_sum(
        value,
        threads_in_warp=16,
    )
    inclusive_sum_out[tidx] = coop._warp.inclusive_sum(
        value,
        threads_in_warp=16,
    )
    exclusive_max_out[tidx] = coop._warp.exclusive_scan(
        value,
        scan_op="max",
        initial_value=_FLOAT32_LOWEST,
        threads_in_warp=16,
    )
    inclusive_max_out[tidx] = coop._warp.inclusive_scan(
        value,
        scan_op="max",
        threads_in_warp=16,
    )

    warp_sum = coop._warp.sum(value, threads_in_warp=16)
    warp_max = coop._warp.max(value, threads_in_warp=16)
    warp_min = coop._warp.min(value, threads_in_warp=16)
    warp_reduce_max = coop._warp.reduce(
        value,
        binary_op="max",
        threads_in_warp=16,
    )
    if lane_id == 0:
        sum_out[logical_warp_id] = warp_sum
        max_out[logical_warp_id] = warp_max
        min_out[logical_warp_id] = warp_min
        reduce_max_out[logical_warp_id] = warp_reduce_max


@cute.jit
def _run_warp_scan_reduce_float32(
    values_in: cute.Tensor,
    exclusive_sum_out: cute.Tensor,
    inclusive_sum_out: cute.Tensor,
    exclusive_max_out: cute.Tensor,
    inclusive_max_out: cute.Tensor,
    sum_out: cute.Tensor,
    max_out: cute.Tensor,
    min_out: cute.Tensor,
    reduce_max_out: cute.Tensor,
):
    _warp_scan_reduce_float32_kernel(
        values_in,
        exclusive_sum_out,
        inclusive_sum_out,
        exclusive_max_out,
        inclusive_max_out,
        sum_out,
        max_out,
        min_out,
        reduce_max_out,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1))


def test_provider_warp_scan_reduce_runtime_float32_logical_warps():
    cutlass.cuda.initialize_cuda_context()

    threads_in_warp = 16
    block_x = 32
    num_logical_warps = block_x // threads_in_warp
    values_host = torch.arange(block_x, dtype=torch.float32) * 0.25 - 4.0
    values_host[0] = -2048.0
    values_host[16] = -1536.0
    values_in = values_host.cuda()
    exclusive_sum_out = torch.zeros((block_x,), dtype=torch.float32, device="cuda")
    inclusive_sum_out = torch.zeros((block_x,), dtype=torch.float32, device="cuda")
    exclusive_max_out = torch.zeros((block_x,), dtype=torch.float32, device="cuda")
    inclusive_max_out = torch.zeros((block_x,), dtype=torch.float32, device="cuda")
    sum_out = torch.zeros(
        (num_logical_warps,),
        dtype=torch.float32,
        device="cuda",
    )
    max_out = torch.zeros(
        (num_logical_warps,),
        dtype=torch.float32,
        device="cuda",
    )
    min_out = torch.zeros(
        (num_logical_warps,),
        dtype=torch.float32,
        device="cuda",
    )
    reduce_max_out = torch.zeros(
        (num_logical_warps,),
        dtype=torch.float32,
        device="cuda",
    )

    _run_warp_scan_reduce_float32(
        from_dlpack(values_in),
        from_dlpack(exclusive_sum_out),
        from_dlpack(inclusive_sum_out),
        from_dlpack(exclusive_max_out),
        from_dlpack(inclusive_max_out),
        from_dlpack(sum_out),
        from_dlpack(max_out),
        from_dlpack(min_out),
        from_dlpack(reduce_max_out),
    )
    torch.cuda.synchronize()

    expected_exclusive_sum = torch.empty_like(values_host)
    expected_inclusive_sum = torch.empty_like(values_host)
    expected_exclusive_max = torch.empty_like(values_host)
    expected_inclusive_max = torch.empty_like(values_host)
    expected_sum = torch.empty((num_logical_warps,), dtype=torch.float32)
    expected_max = torch.empty((num_logical_warps,), dtype=torch.float32)
    expected_min = torch.empty((num_logical_warps,), dtype=torch.float32)
    initial_max = torch.tensor(_FLOAT32_LOWEST, dtype=torch.float32)
    for group_idx, group_base in enumerate(range(0, block_x, threads_in_warp)):
        group = values_host[group_base : group_base + threads_in_warp]
        inclusive_sum = torch.cumsum(group, dim=0)
        inclusive_max = torch.cummax(group, dim=0).values
        expected_exclusive_sum[group_base : group_base + threads_in_warp] = (
            inclusive_sum - group
        )
        expected_inclusive_sum[group_base : group_base + threads_in_warp] = (
            inclusive_sum
        )
        expected_exclusive_max[group_base] = initial_max
        expected_exclusive_max[group_base + 1 : group_base + threads_in_warp] = (
            torch.maximum(initial_max, inclusive_max[:-1])
        )
        expected_inclusive_max[group_base : group_base + threads_in_warp] = (
            inclusive_max
        )
        expected_sum[group_idx] = group.sum()
        expected_max[group_idx] = group.max()
        expected_min[group_idx] = group.min()

    torch.testing.assert_close(
        exclusive_sum_out.cpu(), expected_exclusive_sum, atol=1.0e-6, rtol=0
    )
    torch.testing.assert_close(
        inclusive_sum_out.cpu(), expected_inclusive_sum, atol=1.0e-6, rtol=0
    )
    torch.testing.assert_close(
        exclusive_max_out.cpu(), expected_exclusive_max, atol=0, rtol=0
    )
    torch.testing.assert_close(
        inclusive_max_out.cpu(), expected_inclusive_max, atol=0, rtol=0
    )
    torch.testing.assert_close(sum_out.cpu(), expected_sum, atol=1.0e-6, rtol=0)
    torch.testing.assert_close(max_out.cpu(), expected_max, atol=0, rtol=0)
    torch.testing.assert_close(min_out.cpu(), expected_min, atol=0, rtol=0)
    torch.testing.assert_close(reduce_max_out.cpu(), expected_max, atol=0, rtol=0)


@cute.kernel
def _logical_warp_thread_data_reduce_kernel(
    values_in: cute.Tensor,
    full_out: cute.Tensor,
    partial_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    logical_warp_id = tidx // Int32(16)
    lane_id = tidx - logical_warp_id * Int32(16)
    items = coop.ThreadData.from_values(
        values_in[tidx],
        values_in[tidx + Int32(32)],
        dtype=Int32,
    )
    full = coop._warp.sum(items, threads_in_warp=16)
    partial = coop._warp.sum(
        items,
        threads_in_warp=16,
        valid_items=12,
    )
    if lane_id == 0:
        full_out[logical_warp_id] = full[0]
        partial_out[logical_warp_id] = partial[0]


@cute.jit
def _run_logical_warp_thread_data_reduce(
    values_in: cute.Tensor,
    full_out: cute.Tensor,
    partial_out: cute.Tensor,
):
    _logical_warp_thread_data_reduce_kernel(
        values_in,
        full_out,
        partial_out,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1))


def test_provider_logical_warp_thread_data_reduce_full_and_partial_runtime():
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.arange(1, 65, dtype=torch.int32)
    values_in = values_host.cuda()
    full_out = torch.zeros((2,), dtype=torch.int32, device="cuda")
    partial_out = torch.zeros((2,), dtype=torch.int32, device="cuda")

    _run_logical_warp_thread_data_reduce(
        from_dlpack(values_in),
        from_dlpack(full_out),
        from_dlpack(partial_out),
    )
    torch.cuda.synchronize()

    expected_full = []
    expected_partial = []
    for logical_warp_id in range(2):
        lane_base = logical_warp_id * 16
        full_indices = [
            *range(lane_base, lane_base + 16),
            *range(lane_base + 32, lane_base + 48),
        ]
        partial_indices = [
            *range(lane_base, lane_base + 12),
            *range(lane_base + 32, lane_base + 44),
        ]
        expected_full.append(int(values_host[full_indices].sum().item()))
        expected_partial.append(int(values_host[partial_indices].sum().item()))

    torch.testing.assert_close(
        full_out.cpu(),
        torch.tensor(expected_full, dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        partial_out.cpu(),
        torch.tensor(expected_partial, dtype=torch.int32),
        atol=0,
        rtol=0,
    )
