# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import operator

import pytest

from ..support.runtime import (
    LAUNCH_CASES as _LAUNCH_CASES,
)
from ..support.runtime import (
    SCAN_SUM_TEMP_STORAGE as _SCAN_SUM_TEMP_STORAGE,
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
def _scan_sum_kernel(
    values_in: cute.Tensor,
    prefix_out: cute.Tensor,
    scan_prefix_out: cute.Tensor,
    inclusive_out: cute.Tensor,
    scan_inclusive_out: cute.Tensor,
    total_out: cute.Tensor,
    reduced_out: cute.Tensor,
    max_out: cute.Tensor,
    scan_exclusive_max_out: cute.Tensor,
    scan_inclusive_max_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]
    prefix = coop._block.exclusive_sum(value)
    scan_prefix = coop._block.exclusive_scan(value)
    inclusive = coop._block.inclusive_sum(value)
    scan_inclusive = coop._block.inclusive_scan(value)
    total = coop._block.sum(value)
    reduced = coop._block.reduce(value)
    max_value = coop._block.reduce(value, binary_op="max")
    scan_exclusive_max = coop._block.exclusive_scan(
        value, scan_op="max", initial_value=0
    )
    scan_inclusive_max = coop._block.inclusive_scan(value, scan_op="max")
    prefix_out[tidx] = prefix
    scan_prefix_out[tidx] = scan_prefix
    inclusive_out[tidx] = inclusive
    scan_inclusive_out[tidx] = scan_inclusive
    total_out[tidx] = total
    reduced_out[tidx] = reduced
    max_out[tidx] = max_value
    scan_exclusive_max_out[tidx] = scan_exclusive_max
    scan_inclusive_max_out[tidx] = scan_inclusive_max


@cute.kernel
def _scan_sum_temp_kernel(
    values_in: cute.Tensor,
    prefix_out: cute.Tensor,
    scan_prefix_out: cute.Tensor,
    inclusive_out: cute.Tensor,
    scan_inclusive_out: cute.Tensor,
    total_out: cute.Tensor,
    reduced_out: cute.Tensor,
    max_out: cute.Tensor,
    scan_exclusive_max_out: cute.Tensor,
    scan_inclusive_max_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]
    prefix = coop._block.exclusive_sum(value, temp_storage=_SCAN_SUM_TEMP_STORAGE)
    scan_prefix = coop._block.exclusive_scan(value, temp_storage=_SCAN_SUM_TEMP_STORAGE)
    inclusive = coop._block.inclusive_sum(value, temp_storage=_SCAN_SUM_TEMP_STORAGE)
    scan_inclusive = coop._block.inclusive_scan(
        value, temp_storage=_SCAN_SUM_TEMP_STORAGE
    )
    total = coop._block.sum(value, temp_storage=_SCAN_SUM_TEMP_STORAGE)
    reduced = coop._block.reduce(value, temp_storage=_SCAN_SUM_TEMP_STORAGE)
    max_value = coop._block.reduce(
        value,
        binary_op="max",
        temp_storage=_SCAN_SUM_TEMP_STORAGE,
    )
    scan_exclusive_max = coop._block.exclusive_scan(
        value,
        scan_op="max",
        initial_value=0,
        temp_storage=_SCAN_SUM_TEMP_STORAGE,
    )
    scan_inclusive_max = coop._block.inclusive_scan(
        value,
        scan_op="max",
        temp_storage=_SCAN_SUM_TEMP_STORAGE,
    )
    prefix_out[tidx] = prefix
    scan_prefix_out[tidx] = scan_prefix
    inclusive_out[tidx] = inclusive
    scan_inclusive_out[tidx] = scan_inclusive
    total_out[tidx] = total
    reduced_out[tidx] = reduced
    max_out[tidx] = max_value
    scan_exclusive_max_out[tidx] = scan_exclusive_max
    scan_inclusive_max_out[tidx] = scan_inclusive_max


@cute.jit
def _run_scan_sum(
    values_in: cute.Tensor,
    prefix_out: cute.Tensor,
    scan_prefix_out: cute.Tensor,
    inclusive_out: cute.Tensor,
    scan_inclusive_out: cute.Tensor,
    total_out: cute.Tensor,
    reduced_out: cute.Tensor,
    max_out: cute.Tensor,
    scan_exclusive_max_out: cute.Tensor,
    scan_inclusive_max_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _scan_sum_kernel(
        values_in,
        prefix_out,
        scan_prefix_out,
        inclusive_out,
        scan_inclusive_out,
        total_out,
        reduced_out,
        max_out,
        scan_exclusive_max_out,
        scan_inclusive_max_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_scan_sum_temp(
    values_in: cute.Tensor,
    prefix_out: cute.Tensor,
    scan_prefix_out: cute.Tensor,
    inclusive_out: cute.Tensor,
    scan_inclusive_out: cute.Tensor,
    total_out: cute.Tensor,
    reduced_out: cute.Tensor,
    max_out: cute.Tensor,
    scan_exclusive_max_out: cute.Tensor,
    scan_inclusive_max_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _scan_sum_temp_kernel(
        values_in,
        prefix_out,
        scan_prefix_out,
        inclusive_out,
        scan_inclusive_out,
        total_out,
        reduced_out,
        max_out,
        scan_exclusive_max_out,
        scan_inclusive_max_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_scan_reduce_runtime(block_x: int, use_temp_storage: bool):
    cutlass.cuda.initialize_cuda_context()
    _SCAN_SUM_TEMP_STORAGE.reset_uses()

    values_host = torch.arange(block_x, dtype=torch.int32)
    values_in = values_host.cuda()
    prefix_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    scan_prefix_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    inclusive_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    scan_inclusive_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    total_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    reduced_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    max_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    scan_exclusive_max_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    scan_inclusive_max_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_scan_sum_temp(
            from_dlpack(values_in),
            from_dlpack(prefix_out),
            from_dlpack(scan_prefix_out),
            from_dlpack(inclusive_out),
            from_dlpack(scan_inclusive_out),
            from_dlpack(total_out),
            from_dlpack(reduced_out),
            from_dlpack(max_out),
            from_dlpack(scan_exclusive_max_out),
            from_dlpack(scan_inclusive_max_out),
            block_x,
        )
    else:
        _run_scan_sum(
            from_dlpack(values_in),
            from_dlpack(prefix_out),
            from_dlpack(scan_prefix_out),
            from_dlpack(inclusive_out),
            from_dlpack(scan_inclusive_out),
            from_dlpack(total_out),
            from_dlpack(reduced_out),
            from_dlpack(max_out),
            from_dlpack(scan_exclusive_max_out),
            from_dlpack(scan_inclusive_max_out),
            block_x,
        )
    torch.cuda.synchronize()

    accum = torch.cumsum(values_host.to(torch.int64), dim=0)
    expected_prefix = (accum - values_host.to(torch.int64)).to(torch.int32)
    expected_inclusive = accum.to(torch.int32)
    expected_total = torch.full(
        (block_x,), int(values_host.to(torch.int64).sum().item()), dtype=torch.int32
    )
    expected_max = torch.full(
        (block_x,), int(values_host.max().item()), dtype=torch.int32
    )
    expected_inclusive_max = torch.cummax(values_host, dim=0).values
    expected_exclusive_max = torch.empty_like(values_host)
    expected_exclusive_max[0] = 0
    expected_exclusive_max[1:] = expected_inclusive_max[:-1]

    torch.testing.assert_close(prefix_out.cpu(), expected_prefix, atol=0, rtol=0)
    torch.testing.assert_close(scan_prefix_out.cpu(), expected_prefix, atol=0, rtol=0)
    torch.testing.assert_close(inclusive_out.cpu(), expected_inclusive, atol=0, rtol=0)
    torch.testing.assert_close(
        scan_inclusive_out.cpu(), expected_inclusive, atol=0, rtol=0
    )
    torch.testing.assert_close(total_out.cpu(), expected_total, atol=0, rtol=0)
    torch.testing.assert_close(reduced_out.cpu(), expected_total, atol=0, rtol=0)
    torch.testing.assert_close(max_out.cpu(), expected_max, atol=0, rtol=0)
    torch.testing.assert_close(
        scan_exclusive_max_out.cpu(),
        expected_exclusive_max,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        scan_inclusive_max_out.cpu(),
        expected_inclusive_max,
        atol=0,
        rtol=0,
    )


@cute.kernel
def _group_scan_parity_kernel(
    values_in: cute.Tensor,
    block_root_out: cute.Tensor,
    block_scoped_out: cute.Tensor,
    block_aggregate_out: cute.Tensor,
    warp_root_out: cute.Tensor,
    warp_scoped_out: cute.Tensor,
    warp_aggregate_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]

    block_root_out[tidx] = coop.scan(coop.this_block(), value)
    block_scoped_out[tidx] = coop._block.exclusive_sum(value)
    block_aggregate = coop.ThreadData(1, dtype=Int32)
    coop._block.exclusive_sum(value, block_aggregate=block_aggregate)
    block_aggregate_out[tidx] = block_aggregate[0]

    warp_root_out[tidx] = coop.scan(coop.this_warp(), value)
    warp_scoped_out[tidx] = coop._warp.exclusive_sum(value)
    warp_aggregate = coop.ThreadData(1, dtype=Int32)
    coop._warp.exclusive_sum(value, warp_aggregate=warp_aggregate)
    warp_aggregate_out[tidx] = warp_aggregate[0]


@cute.jit
def _run_group_scan_parity(
    values_in: cute.Tensor,
    block_root_out: cute.Tensor,
    block_scoped_out: cute.Tensor,
    block_aggregate_out: cute.Tensor,
    warp_root_out: cute.Tensor,
    warp_scoped_out: cute.Tensor,
    warp_aggregate_out: cute.Tensor,
):
    _group_scan_parity_kernel(
        values_in,
        block_root_out,
        block_scoped_out,
        block_aggregate_out,
        warp_root_out,
        warp_scoped_out,
        warp_aggregate_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


def test_provider_group_scan_root_scoped_and_aggregate_runtime():
    cutlass.cuda.initialize_cuda_context()

    block_x = 64
    values_host = (torch.arange(block_x, dtype=torch.int32) % 13) + 1
    values_in = values_host.cuda()
    outputs = [
        torch.zeros((block_x,), dtype=torch.int32, device="cuda") for _ in range(6)
    ]
    _run_group_scan_parity(
        from_dlpack(values_in),
        *(from_dlpack(output) for output in outputs),
    )
    torch.cuda.synchronize()

    block_inclusive = torch.cumsum(values_host.to(torch.int64), dim=0).to(torch.int32)
    expected_block = block_inclusive - values_host
    expected_block_aggregate = torch.full_like(values_host, values_host.sum())
    expected_warp = torch.empty_like(values_host)
    expected_warp_aggregate = torch.empty_like(values_host)
    for base in range(0, block_x, 32):
        warp_values = values_host[base : base + 32]
        warp_inclusive = torch.cumsum(warp_values.to(torch.int64), dim=0).to(
            torch.int32
        )
        expected_warp[base : base + 32] = warp_inclusive - warp_values
        expected_warp_aggregate[base : base + 32] = warp_values.sum()

    (
        block_root,
        block_scoped,
        block_aggregate,
        warp_root,
        warp_scoped,
        warp_aggregate,
    ) = (output.cpu() for output in outputs)
    torch.testing.assert_close(block_root, expected_block, atol=0, rtol=0)
    torch.testing.assert_close(block_scoped, expected_block, atol=0, rtol=0)
    torch.testing.assert_close(
        block_aggregate,
        expected_block_aggregate,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(warp_root, expected_warp, atol=0, rtol=0)
    torch.testing.assert_close(warp_scoped, expected_warp, atol=0, rtol=0)
    torch.testing.assert_close(
        warp_aggregate,
        expected_warp_aggregate,
        atol=0,
        rtol=0,
    )


@cute.kernel
def _root_block_scan_thread_data_kernel(
    values_in: cute.Tensor,
    one_item_out: cute.Tensor,
    four_items_out: cute.Tensor,
    outbound_ssa_out: cute.Tensor,
    outbound_rmem_out: cute.Tensor,
    rmem_out: cute.Tensor,
    tensorssa_out: cute.Tensor,
    scoped_tensorssa_out: cute.Tensor,
    algorithm: cutlass.Constexpr[str],
):
    tx, ty, tz = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_dim()
    linear_tid = tx + bx * (ty + by * tz)

    one_item = coop.ThreadData.from_values(values_in[linear_tid], dtype=Int32)
    one_item_result = coop.scan(
        coop.this_block(),
        one_item,
        algorithm=algorithm,
    )
    one_item_out[linear_tid] = one_item_result[0]

    four_items = coop._block.load(values_in, items_per_thread=4, dtype=Int32)
    four_items_result = coop.scan(
        coop.this_block(),
        four_items,
        algorithm=algorithm,
    )
    coop._block.store(four_items_out, four_items_result)
    outbound_ssa = four_items_result.to_tensor_ssa(shape=(2, 2))
    outbound_rmem = four_items_result.to_register_tensor(shape=(2, 2))
    coop._block.store(outbound_ssa_out, outbound_ssa)
    coop._block.store(outbound_rmem_out, outbound_rmem)

    fragment_base = linear_tid * 2
    fragment = cute.make_rmem_tensor((1, 2), Int32)
    fragment[0] = values_in[fragment_base]
    fragment[1] = values_in[fragment_base + 1]
    rmem_result = coop.scan(
        coop.this_block(),
        fragment,
        algorithm=algorithm,
    )
    ssa = fragment.load()
    tensorssa_result = coop.scan(
        coop.this_block(),
        ssa,
        algorithm=algorithm,
    )
    scoped_tensorssa_result = coop._block.exclusive_sum(ssa)
    rmem_out[fragment_base] = rmem_result[0]
    rmem_out[fragment_base + 1] = rmem_result[1]
    tensorssa_out[fragment_base] = tensorssa_result[0]
    tensorssa_out[fragment_base + 1] = tensorssa_result[1]
    scoped_tensorssa_out[fragment_base] = scoped_tensorssa_result[0]
    scoped_tensorssa_out[fragment_base + 1] = scoped_tensorssa_result[1]


@cute.jit
def _run_root_block_scan_thread_data(
    values_in: cute.Tensor,
    one_item_out: cute.Tensor,
    four_items_out: cute.Tensor,
    outbound_ssa_out: cute.Tensor,
    outbound_rmem_out: cute.Tensor,
    rmem_out: cute.Tensor,
    tensorssa_out: cute.Tensor,
    scoped_tensorssa_out: cute.Tensor,
    algorithm: cutlass.Constexpr[str],
    block_x: cutlass.Constexpr,
    block_y: cutlass.Constexpr,
    block_z: cutlass.Constexpr,
):
    _root_block_scan_thread_data_kernel(
        values_in,
        one_item_out,
        four_items_out,
        outbound_ssa_out,
        outbound_rmem_out,
        rmem_out,
        tensorssa_out,
        scoped_tensorssa_out,
        algorithm,
    ).launch(grid=(1, 1, 1), block=(block_x, block_y, block_z))


@pytest.mark.parametrize(
    "algorithm,block_dim",
    [
        ("raking", (64, 1, 1)),
        ("raking_memoize", (8, 4, 2)),
        ("warp_scans", (16, 4, 1)),
    ],
    ids=["raking-1d", "raking-memoize-3d", "warp-scans-2d"],
)
def test_provider_root_block_scan_thread_data_output_abi_runtime(
    algorithm: str,
    block_dim: tuple[int, int, int],
):
    cutlass.cuda.initialize_cuda_context()

    block_threads = block_dim[0] * block_dim[1] * block_dim[2]
    items_per_thread = 4
    values_host = (
        torch.arange(block_threads * items_per_thread, dtype=torch.int32) % 11
    ) + 1
    values_in = values_host.cuda()
    one_item_out = torch.zeros(
        (block_threads,),
        dtype=torch.int32,
        device="cuda",
    )
    four_items_out = torch.zeros_like(values_in)
    outbound_outputs = [torch.zeros_like(values_in) for _ in range(2)]
    two_item_outputs = [
        torch.zeros((block_threads * 2,), dtype=torch.int32, device="cuda")
        for _ in range(3)
    ]

    _run_root_block_scan_thread_data(
        from_dlpack(values_in),
        from_dlpack(one_item_out),
        from_dlpack(four_items_out),
        *(from_dlpack(output) for output in outbound_outputs),
        *(from_dlpack(output) for output in two_item_outputs),
        algorithm,
        *block_dim,
    )
    torch.cuda.synchronize()

    one_item_values = values_host[:block_threads]
    one_item_inclusive = torch.cumsum(one_item_values.to(torch.int64), dim=0).to(
        torch.int32
    )
    four_items_inclusive = torch.cumsum(values_host.to(torch.int64), dim=0).to(
        torch.int32
    )
    two_item_values = values_host[: block_threads * 2]
    two_items_inclusive = torch.cumsum(two_item_values.to(torch.int64), dim=0).to(
        torch.int32
    )
    expected_two_items = two_items_inclusive - two_item_values
    torch.testing.assert_close(
        one_item_out.cpu(),
        one_item_inclusive - one_item_values,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        four_items_out.cpu(),
        four_items_inclusive - values_host,
        atol=0,
        rtol=0,
    )
    for output in outbound_outputs:
        torch.testing.assert_close(
            output.cpu(),
            four_items_inclusive - values_host,
            atol=0,
            rtol=0,
        )
    for output in two_item_outputs:
        torch.testing.assert_close(
            output.cpu(),
            expected_two_items,
            atol=0,
            rtol=0,
        )


@cute.kernel
def _scan_custom_initial_aggregate_kernel(
    values_in: cute.Tensor,
    initial_in: cute.Tensor,
    block_prefix_out: cute.Tensor,
    block_aggregate_out: cute.Tensor,
    warp_prefix_out: cute.Tensor,
    warp_aggregate_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]
    initial_value = initial_in[0]

    block_aggregate = coop.ThreadData(1, dtype=Int32)
    block_prefix_out[tidx] = coop._block.exclusive_scan(
        value,
        scan_op="max",
        initial_value=initial_value,
        block_aggregate=block_aggregate,
    )
    block_aggregate_out[tidx] = block_aggregate[0]

    warp_aggregate = coop.ThreadData(1, dtype=Int32)
    warp_prefix_out[tidx] = coop._warp.exclusive_scan(
        value,
        scan_op="max",
        initial_value=initial_value,
        warp_aggregate=warp_aggregate,
    )
    warp_aggregate_out[tidx] = warp_aggregate[0]


@cute.jit
def _run_scan_custom_initial_aggregate(
    values_in: cute.Tensor,
    initial_in: cute.Tensor,
    block_prefix_out: cute.Tensor,
    block_aggregate_out: cute.Tensor,
    warp_prefix_out: cute.Tensor,
    warp_aggregate_out: cute.Tensor,
):
    _scan_custom_initial_aggregate_kernel(
        values_in,
        initial_in,
        block_prefix_out,
        block_aggregate_out,
        warp_prefix_out,
        warp_aggregate_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


def test_provider_scoped_scan_runtime_initial_and_aggregate_exclusion():
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.cat(
        (
            torch.arange(1, 33, dtype=torch.int32),
            torch.arange(101, 133, dtype=torch.int32),
        )
    )
    initial_host = torch.tensor([1000], dtype=torch.int32)
    values_in = values_host.cuda()
    initial_in = initial_host.cuda()
    outputs = [
        torch.zeros_like(values_in),
        torch.zeros_like(values_in),
        torch.zeros_like(values_in),
        torch.zeros_like(values_in),
    ]

    _run_scan_custom_initial_aggregate(
        from_dlpack(values_in),
        from_dlpack(initial_in),
        *(from_dlpack(output) for output in outputs),
    )
    torch.cuda.synchronize()

    expected_prefix = torch.full_like(values_host, initial_host.item())
    expected_block_aggregate = torch.full_like(values_host, values_host.max())
    expected_warp_aggregate = torch.cat(
        (
            torch.full((32,), values_host[:32].max(), dtype=torch.int32),
            torch.full((32,), values_host[32:].max(), dtype=torch.int32),
        )
    )
    block_prefix, block_aggregate, warp_prefix, warp_aggregate = (
        output.cpu() for output in outputs
    )
    torch.testing.assert_close(block_prefix, expected_prefix, atol=0, rtol=0)
    torch.testing.assert_close(
        block_aggregate,
        expected_block_aggregate,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(warp_prefix, expected_prefix, atol=0, rtol=0)
    torch.testing.assert_close(
        warp_aggregate,
        expected_warp_aggregate,
        atol=0,
        rtol=0,
    )


@cute.kernel
def _scan_reduce_callable_op_kernel(
    values_in: cute.Tensor,
    exclusive_xor_out: cute.Tensor,
    inclusive_xor_out: cute.Tensor,
    reduced_xor_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]
    exclusive_xor = coop._block.exclusive_scan(
        value,
        scan_op=operator.xor,
        initial_value=0,
        temp_storage=_SCAN_SUM_TEMP_STORAGE,
    )
    inclusive_xor = coop._block.inclusive_scan(
        value,
        scan_op=operator.xor,
        temp_storage=_SCAN_SUM_TEMP_STORAGE,
    )
    reduced_xor = coop._block.reduce(
        value,
        binary_op=operator.xor,
        temp_storage=_SCAN_SUM_TEMP_STORAGE,
    )
    exclusive_xor_out[tidx] = exclusive_xor
    inclusive_xor_out[tidx] = inclusive_xor
    reduced_xor_out[tidx] = reduced_xor


@cute.jit
def _run_scan_reduce_callable_op(
    values_in: cute.Tensor,
    exclusive_xor_out: cute.Tensor,
    inclusive_xor_out: cute.Tensor,
    reduced_xor_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _scan_reduce_callable_op_kernel(
        values_in,
        exclusive_xor_out,
        inclusive_xor_out,
        reduced_xor_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


def test_provider_scan_reduce_runtime_known_callable_operator_aliases():
    cutlass.cuda.initialize_cuda_context()
    _SCAN_SUM_TEMP_STORAGE.reset_uses()

    block_x = 64
    values_host = torch.arange(block_x, dtype=torch.int32)
    values_in = values_host.cuda()
    exclusive_xor_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    inclusive_xor_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    reduced_xor_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    _run_scan_reduce_callable_op(
        from_dlpack(values_in),
        from_dlpack(exclusive_xor_out),
        from_dlpack(inclusive_xor_out),
        from_dlpack(reduced_xor_out),
        block_x,
    )
    torch.cuda.synchronize()

    expected_exclusive = torch.empty_like(values_host)
    expected_inclusive = torch.empty_like(values_host)
    running = 0
    for idx, value in enumerate(values_host.tolist()):
        expected_exclusive[idx] = running
        running ^= value
        expected_inclusive[idx] = running
    expected_reduced = torch.full((block_x,), running, dtype=torch.int32)

    torch.testing.assert_close(
        exclusive_xor_out.cpu(), expected_exclusive, atol=0, rtol=0
    )
    torch.testing.assert_close(
        inclusive_xor_out.cpu(), expected_inclusive, atol=0, rtol=0
    )
    torch.testing.assert_close(reduced_xor_out.cpu(), expected_reduced, atol=0, rtol=0)


@cute.kernel
def _scan_thread_data_kernel(
    values_in: cute.Tensor,
    exclusive_sum_out: cute.Tensor,
    inclusive_sum_out: cute.Tensor,
    exclusive_xor_out: cute.Tensor,
    inclusive_xor_out: cute.Tensor,
):
    items = coop._block.load(values_in, items_per_thread=4, dtype=Int32)
    exclusive_sum_items = coop._block.exclusive_sum(items)
    inclusive_sum_items = coop._block.inclusive_sum(items)
    exclusive_xor_items = coop._block.exclusive_scan(
        items,
        scan_op="bit_xor",
        initial_value=0,
    )
    inclusive_xor_items = coop._block.inclusive_scan(items, scan_op="bit_xor")
    coop._block.store(exclusive_sum_out, exclusive_sum_items)
    coop._block.store(inclusive_sum_out, inclusive_sum_items)
    coop._block.store(exclusive_xor_out, exclusive_xor_items)
    coop._block.store(inclusive_xor_out, inclusive_xor_items)


@cute.kernel
def _scan_thread_data_temp_kernel(
    values_in: cute.Tensor,
    exclusive_sum_out: cute.Tensor,
    inclusive_sum_out: cute.Tensor,
    exclusive_xor_out: cute.Tensor,
    inclusive_xor_out: cute.Tensor,
):
    items = coop._block.load(values_in, items_per_thread=4, dtype=Int32)
    exclusive_sum_items = coop._block.exclusive_sum(
        items,
        temp_storage=_SCAN_SUM_TEMP_STORAGE,
    )
    inclusive_sum_items = coop._block.inclusive_sum(
        items,
        temp_storage=_SCAN_SUM_TEMP_STORAGE,
    )
    exclusive_xor_items = coop._block.exclusive_scan(
        items,
        scan_op="bit_xor",
        initial_value=0,
        temp_storage=_SCAN_SUM_TEMP_STORAGE,
    )
    inclusive_xor_items = coop._block.inclusive_scan(
        items,
        scan_op="bit_xor",
        temp_storage=_SCAN_SUM_TEMP_STORAGE,
    )
    coop._block.store(exclusive_sum_out, exclusive_sum_items)
    coop._block.store(inclusive_sum_out, inclusive_sum_items)
    coop._block.store(exclusive_xor_out, exclusive_xor_items)
    coop._block.store(inclusive_xor_out, inclusive_xor_items)


@cute.jit
def _run_scan_thread_data(
    values_in: cute.Tensor,
    exclusive_sum_out: cute.Tensor,
    inclusive_sum_out: cute.Tensor,
    exclusive_xor_out: cute.Tensor,
    inclusive_xor_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _scan_thread_data_kernel(
        values_in,
        exclusive_sum_out,
        inclusive_sum_out,
        exclusive_xor_out,
        inclusive_xor_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_scan_thread_data_temp(
    values_in: cute.Tensor,
    exclusive_sum_out: cute.Tensor,
    inclusive_sum_out: cute.Tensor,
    exclusive_xor_out: cute.Tensor,
    inclusive_xor_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _scan_thread_data_temp_kernel(
        values_in,
        exclusive_sum_out,
        inclusive_sum_out,
        exclusive_xor_out,
        inclusive_xor_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_scan_runtime_thread_data_multi_item(
    block_x: int,
    use_temp_storage: bool,
):
    cutlass.cuda.initialize_cuda_context()
    _SCAN_SUM_TEMP_STORAGE.reset_uses()

    items_per_thread = 4
    total_items = block_x * items_per_thread
    values_host = (torch.arange(total_items, dtype=torch.int32) % 17) + 1
    values_in = values_host.cuda()
    exclusive_sum_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    inclusive_sum_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    exclusive_xor_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    inclusive_xor_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_scan_thread_data_temp(
            from_dlpack(values_in),
            from_dlpack(exclusive_sum_out),
            from_dlpack(inclusive_sum_out),
            from_dlpack(exclusive_xor_out),
            from_dlpack(inclusive_xor_out),
            block_x,
        )
    else:
        _run_scan_thread_data(
            from_dlpack(values_in),
            from_dlpack(exclusive_sum_out),
            from_dlpack(inclusive_sum_out),
            from_dlpack(exclusive_xor_out),
            from_dlpack(inclusive_xor_out),
            block_x,
        )
    torch.cuda.synchronize()

    inclusive_sum_expected = torch.cumsum(values_host.to(torch.int64), dim=0).to(
        torch.int32
    )
    exclusive_sum_expected = inclusive_sum_expected - values_host
    exclusive_xor_expected = torch.empty_like(values_host)
    inclusive_xor_expected = torch.empty_like(values_host)
    running = 0
    for idx, value in enumerate(values_host.tolist()):
        exclusive_xor_expected[idx] = running
        running ^= value
        inclusive_xor_expected[idx] = running

    torch.testing.assert_close(
        exclusive_sum_out.cpu(), exclusive_sum_expected, atol=0, rtol=0
    )
    torch.testing.assert_close(
        inclusive_sum_out.cpu(), inclusive_sum_expected, atol=0, rtol=0
    )
    torch.testing.assert_close(
        exclusive_xor_out.cpu(), exclusive_xor_expected, atol=0, rtol=0
    )
    torch.testing.assert_close(
        inclusive_xor_out.cpu(), inclusive_xor_expected, atol=0, rtol=0
    )
