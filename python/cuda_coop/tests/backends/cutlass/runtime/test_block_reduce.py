# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.runtime import (
    GROUP_REDUCE_COMPAT_STORAGE as _GROUP_REDUCE_COMPAT_STORAGE,
)
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
def _reduce_thread_data_kernel(
    values_in: cute.Tensor,
    sum_out: cute.Tensor,
    xor_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    items = coop._block.load(values_in, items_per_thread=4, dtype=Int32)
    total = coop._block.sum(items)
    xor_total = coop._block.reduce(items, binary_op="bit_xor")
    sum_out[tidx] = total
    xor_out[tidx] = xor_total


@cute.kernel
def _reduce_thread_data_temp_kernel(
    values_in: cute.Tensor,
    sum_out: cute.Tensor,
    xor_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    items = coop._block.load(values_in, items_per_thread=4, dtype=Int32)
    total = coop._block.sum(items, temp_storage=_SCAN_SUM_TEMP_STORAGE)
    xor_total = coop._block.reduce(
        items,
        binary_op="bit_xor",
        temp_storage=_SCAN_SUM_TEMP_STORAGE,
    )
    sum_out[tidx] = total
    xor_out[tidx] = xor_total


@cute.jit
def _run_reduce_thread_data(
    values_in: cute.Tensor,
    sum_out: cute.Tensor,
    xor_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _reduce_thread_data_kernel(values_in, sum_out, xor_out).launch(
        grid=(1, 1, 1), block=(block_x, 1, 1)
    )


@cute.jit
def _run_reduce_thread_data_temp(
    values_in: cute.Tensor,
    sum_out: cute.Tensor,
    xor_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _reduce_thread_data_temp_kernel(values_in, sum_out, xor_out).launch(
        grid=(1, 1, 1), block=(block_x, 1, 1)
    )


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_reduce_runtime_thread_data_multi_item(
    block_x: int,
    use_temp_storage: bool,
):
    cutlass.cuda.initialize_cuda_context()
    _SCAN_SUM_TEMP_STORAGE.reset_uses()

    items_per_thread = 4
    total_items = block_x * items_per_thread
    values_host = (torch.arange(total_items, dtype=torch.int32) % 17) + 1
    values_in = values_host.cuda()
    sum_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    xor_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_reduce_thread_data_temp(
            from_dlpack(values_in),
            from_dlpack(sum_out),
            from_dlpack(xor_out),
            block_x,
        )
    else:
        _run_reduce_thread_data(
            from_dlpack(values_in),
            from_dlpack(sum_out),
            from_dlpack(xor_out),
            block_x,
        )
    torch.cuda.synchronize()

    running_xor = 0
    for value in values_host.tolist():
        running_xor ^= value
    expected_sum = torch.full(
        (block_x,),
        int(values_host.to(torch.int64).sum().item()),
        dtype=torch.int32,
    )
    expected_xor = torch.full((block_x,), running_xor, dtype=torch.int32)

    torch.testing.assert_close(sum_out.cpu(), expected_sum, atol=0, rtol=0)
    torch.testing.assert_close(xor_out.cpu(), expected_xor, atol=0, rtol=0)


@cute.kernel
def _cudax_thread_group_reduce_kernel(
    values_in: cute.Tensor,
    block_sum_out: cute.Tensor,
    block_items_sum_out: cute.Tensor,
    block_rmem_sum_out: cute.Tensor,
    block_tensorssa_sum_out: cute.Tensor,
    warp_max_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    block_group = coop.this_block()
    warp_group = coop.this_warp()

    value = values_in[tidx]
    block_sum_out[tidx] = coop.reduce(block_group, value)

    item_base = tidx * 2
    items = coop.ThreadData.from_values(
        values_in[item_base],
        values_in[item_base + 1],
        dtype=Int32,
    )
    block_items_sum = coop.reduce(block_group, items)
    block_items_sum_out[tidx] = block_items_sum

    fragment = cute.make_rmem_tensor((1, 2), Int32)
    fragment[0] = values_in[item_base]
    fragment[1] = values_in[item_base + 1]
    block_rmem_sum_out[tidx] = coop.reduce(block_group, fragment)
    block_tensorssa_sum_out[tidx] = coop.reduce(block_group, fragment.load())

    warp_max_out[tidx] = coop.reduce(warp_group, value, binary_op="max")


@cute.jit
def _run_cudax_thread_group_reduce(
    values_in: cute.Tensor,
    block_sum_out: cute.Tensor,
    block_items_sum_out: cute.Tensor,
    block_rmem_sum_out: cute.Tensor,
    block_tensorssa_sum_out: cute.Tensor,
    warp_max_out: cute.Tensor,
):
    _cudax_thread_group_reduce_kernel(
        values_in,
        block_sum_out,
        block_items_sum_out,
        block_rmem_sum_out,
        block_tensorssa_sum_out,
        warp_max_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


def test_provider_cudax_thread_group_reduce_runtime():
    cutlass.cuda.initialize_cuda_context()

    block_x = 64
    values_host = torch.arange(1, block_x * 2 + 1, dtype=torch.int32)
    values_in = values_host.cuda()
    block_sum_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    block_items_sum_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    block_rmem_sum_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    block_tensorssa_sum_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    warp_max_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    _run_cudax_thread_group_reduce(
        from_dlpack(values_in),
        from_dlpack(block_sum_out),
        from_dlpack(block_items_sum_out),
        from_dlpack(block_rmem_sum_out),
        from_dlpack(block_tensorssa_sum_out),
        from_dlpack(warp_max_out),
    )
    torch.cuda.synchronize()

    expected_block_sum = torch.full(
        (block_x,),
        int(values_host[:block_x].to(torch.int64).sum().item()),
        dtype=torch.int32,
    )
    expected_block_items_sum = torch.full(
        (block_x,),
        int(values_host.to(torch.int64).sum().item()),
        dtype=torch.int32,
    )
    expected_warp_max = torch.cat(
        (
            torch.full((32,), int(values_host[31].item()), dtype=torch.int32),
            torch.full((32,), int(values_host[63].item()), dtype=torch.int32),
        )
    )

    torch.testing.assert_close(block_sum_out.cpu(), expected_block_sum, atol=0, rtol=0)
    torch.testing.assert_close(
        block_items_sum_out.cpu(),
        expected_block_items_sum,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        block_rmem_sum_out.cpu(),
        expected_block_items_sum,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        block_tensorssa_sum_out.cpu(),
        expected_block_items_sum,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(warp_max_out.cpu(), expected_warp_max, atol=0, rtol=0)


_FACTORY_CUB_SUM = None


@cute.kernel
def _mixed_scalar_one_item_reduce_kernel(
    values_in: cute.Tensor,
    cudax_scalar_out: cute.Tensor,
    cudax_one_item_out: cute.Tensor,
    cub_scalar_out: cute.Tensor,
    cub_one_item_out: cute.Tensor,
    cub_valid_default_out: cute.Tensor,
    cub_valid_explicit_out: cute.Tensor,
    factory_cub_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]
    one_item = coop.ThreadData.from_values(value, dtype=Int32)
    cudax_scalar_out[tidx] = coop._block.sum(value)
    cudax_one_item_out[tidx] = coop._block.sum(one_item)
    cub_scalar = coop._block.sum(value, algorithm="warp_reductions")
    cub_one_item = coop._block.sum(
        one_item,
        algorithm="warp_reductions",
    )
    cub_valid_default = coop._block.sum(value, valid_items=17)
    cub_valid_explicit = coop._block.sum(
        value,
        valid_items=17,
        algorithm="warp_reductions",
    )
    factory_cub = _FACTORY_CUB_SUM(value)
    if tidx == 0:
        cub_scalar_out[0] = cub_scalar
        cub_one_item_out[0] = cub_one_item
        cub_valid_default_out[0] = cub_valid_default
        cub_valid_explicit_out[0] = cub_valid_explicit
        factory_cub_out[0] = factory_cub


@cute.jit
def _run_mixed_scalar_one_item_reduce(
    values_in: cute.Tensor,
    cudax_scalar_out: cute.Tensor,
    cudax_one_item_out: cute.Tensor,
    cub_scalar_out: cute.Tensor,
    cub_one_item_out: cute.Tensor,
    cub_valid_default_out: cute.Tensor,
    cub_valid_explicit_out: cute.Tensor,
    factory_cub_out: cute.Tensor,
):
    _mixed_scalar_one_item_reduce_kernel(
        values_in,
        cudax_scalar_out,
        cudax_one_item_out,
        cub_scalar_out,
        cub_one_item_out,
        cub_valid_default_out,
        cub_valid_explicit_out,
        factory_cub_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


def test_provider_mixed_scalar_one_item_reduce_runtime():
    global _FACTORY_CUB_SUM

    cutlass.cuda.initialize_cuda_context()
    _FACTORY_CUB_SUM = coop._block.make_sum(Int32, threads_per_block=64)

    try:
        values_host = torch.arange(1, 65, dtype=torch.int32)
        values_in = values_host.cuda()
        outputs = [
            torch.zeros((64,), dtype=torch.int32, device="cuda") for _ in range(7)
        ]

        _run_mixed_scalar_one_item_reduce(
            from_dlpack(values_in),
            *(from_dlpack(output) for output in outputs),
        )
        torch.cuda.synchronize()

        expected = torch.full((64,), int(values_host.sum().item()), dtype=torch.int32)
        for output in outputs[:2]:
            torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
        for output in outputs[2:4]:
            assert output[0].cpu().item() == expected[0].item()
        expected_valid = int(values_host[:17].sum().item())
        for output in outputs[4:6]:
            assert output[0].cpu().item() == expected_valid
        assert outputs[6][0].cpu().item() == expected[0].item()
    finally:
        _FACTORY_CUB_SUM = None


@cute.kernel
def _floating_reduce_route_equivalence_kernel(
    values_in: cute.Tensor,
    group_out: cute.Tensor,
    scoped_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]
    group_out[tidx] = coop.reduce(coop.this_block(), value)
    scoped_out[tidx] = coop._block.sum(value)


@cute.jit
def _run_floating_reduce_route_equivalence(
    values_in: cute.Tensor,
    group_out: cute.Tensor,
    scoped_out: cute.Tensor,
):
    _floating_reduce_route_equivalence_kernel(
        values_in,
        group_out,
        scoped_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


@pytest.mark.parametrize("case", ("cancellation", "special_values"))
def test_provider_floating_reduce_routes_are_bitwise_identical(case):
    cutlass.cuda.initialize_cuda_context()

    if case == "cancellation":
        values = [1.0e20, 1.0, -1.0e20, -0.0, 3.0, 1.0e-7, -1.0e-7, 0.0]
    else:
        maximum = torch.finfo(torch.float32).max
        values = [float("nan"), float("inf"), -float("inf"), -0.0, 0.0]
        values.extend((maximum, maximum, -maximum))

    # These order- and special-value-sensitive inputs deliberately do not pin
    # the retired reduction tree. They pin root/scoped bit identity for the
    # selected official CUDAX artifact and full-result broadcast. The direct
    # C++ differential gate owns the normative special-value oracle.
    values_host = torch.tensor(values * 8, dtype=torch.float32)
    values_in = values_host.cuda()
    group_out = torch.zeros((64,), dtype=torch.float32, device="cuda")
    scoped_out = torch.zeros((64,), dtype=torch.float32, device="cuda")

    _run_floating_reduce_route_equivalence(
        from_dlpack(values_in),
        from_dlpack(group_out),
        from_dlpack(scoped_out),
    )
    torch.cuda.synchronize()

    group_bits = group_out.cpu().view(torch.int32)
    scoped_bits = scoped_out.cpu().view(torch.int32)
    assert torch.equal(group_bits, scoped_bits)
    assert torch.unique(group_bits).numel() == 1


@cute.kernel
def _full_valid_count_reduce_kernel(
    values_in: cute.Tensor,
    block_static_out: cute.Tensor,
    block_runtime_out: cute.Tensor,
    warp_static_out: cute.Tensor,
    warp_runtime_out: cute.Tensor,
    block_valid_items: Int32,
    warp_valid_items: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]
    block_static = coop.reduce(
        coop.this_block(),
        value,
        broadcast=False,
        valid_items=64,
    )
    block_runtime = coop._block.sum(value, valid_items=block_valid_items)
    warp_static = coop.reduce(
        coop.this_warp(),
        value,
        broadcast=False,
        valid_items=32,
    )
    warp_runtime = coop._warp.sum(
        value,
        threads_in_warp=32,
        valid_items=warp_valid_items,
    )

    if tidx == 0:
        block_static_out[0] = block_static
        block_runtime_out[0] = block_runtime
    warp_id = tidx // Int32(32)
    lane_id = tidx - warp_id * Int32(32)
    if lane_id == 0:
        warp_static_out[warp_id] = warp_static
        warp_runtime_out[warp_id] = warp_runtime


@cute.jit
def _run_full_valid_count_reduce(
    values_in: cute.Tensor,
    block_static_out: cute.Tensor,
    block_runtime_out: cute.Tensor,
    warp_static_out: cute.Tensor,
    warp_runtime_out: cute.Tensor,
    block_valid_items: Int32,
    warp_valid_items: Int32,
):
    _full_valid_count_reduce_kernel(
        values_in,
        block_static_out,
        block_runtime_out,
        warp_static_out,
        warp_runtime_out,
        block_valid_items,
        warp_valid_items,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


def test_provider_full_size_valid_count_remains_root_only_cub():
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.arange(1, 65, dtype=torch.int32)
    values_in = values_host.cuda()
    block_static_out = torch.zeros((1,), dtype=torch.int32, device="cuda")
    block_runtime_out = torch.zeros((1,), dtype=torch.int32, device="cuda")
    warp_static_out = torch.zeros((2,), dtype=torch.int32, device="cuda")
    warp_runtime_out = torch.zeros((2,), dtype=torch.int32, device="cuda")

    _run_full_valid_count_reduce(
        from_dlpack(values_in),
        from_dlpack(block_static_out),
        from_dlpack(block_runtime_out),
        from_dlpack(warp_static_out),
        from_dlpack(warp_runtime_out),
        64,
        32,
    )
    torch.cuda.synchronize()

    expected_block = values_host.sum().to(torch.int32).reshape(1)
    expected_warp = torch.stack((values_host[:32].sum(), values_host[32:].sum())).to(
        torch.int32
    )
    torch.testing.assert_close(block_static_out.cpu(), expected_block, atol=0, rtol=0)
    torch.testing.assert_close(block_runtime_out.cpu(), expected_block, atol=0, rtol=0)
    torch.testing.assert_close(warp_static_out.cpu(), expected_warp, atol=0, rtol=0)
    torch.testing.assert_close(warp_runtime_out.cpu(), expected_warp, atol=0, rtol=0)


@cute.kernel
def _group_reduce_variant_kernel(
    values_in: cute.Tensor,
    block_root_out: cute.Tensor,
    block_root_second_out: cute.Tensor,
    block_partial_out: cute.Tensor,
    block_array_out: cute.Tensor,
    warp_partial_out: cute.Tensor,
    scoped_block_partial_out: cute.Tensor,
    scoped_block_array_out: cute.Tensor,
    scoped_warp_partial_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    block_group = coop.this_block()
    warp_group = coop.this_warp()
    value = values_in[tidx]

    block_root = coop.reduce(block_group, value, broadcast=False)
    block_root_second = coop.reduce(
        block_group,
        values_in[tidx + Int32(64)],
        broadcast=False,
    )
    block_partial = coop.reduce(
        block_group,
        value,
        broadcast=False,
        valid_items=Int32(48),
        algorithm="raking",
    )
    block_items = coop.ThreadData.from_values(
        value,
        values_in[tidx + Int32(64)],
        dtype=Int32,
    )
    block_array = coop.reduce(
        block_group,
        block_items,
        broadcast=False,
        algorithm="warp_reductions",
    )
    warp_partial = coop.reduce(
        warp_group,
        value,
        broadcast=False,
        valid_items=24,
    )
    scoped_block_partial = coop._block.sum(
        value,
        valid_items=Int32(48),
        algorithm="raking",
        temp_storage=_GROUP_REDUCE_COMPAT_STORAGE,
    )
    scoped_block_array = coop._block.sum(
        block_items,
        algorithm="warp_reductions",
    )
    scoped_warp_partial = coop._warp.sum(
        value,
        threads_in_warp=32,
        valid_items=24,
    )

    if tidx == 0:
        block_root_out[0] = block_root
        block_root_second_out[0] = block_root_second
        block_partial_out[0] = block_partial
        block_array_out[0] = block_array
        scoped_block_partial_out[0] = scoped_block_partial
        scoped_block_array_out[0] = scoped_block_array
    warp_id = tidx // Int32(32)
    lane = tidx - warp_id * Int32(32)
    if lane == 0:
        warp_partial_out[warp_id] = warp_partial
        scoped_warp_partial_out[warp_id] = scoped_warp_partial


@cute.jit
def _run_group_reduce_variants(
    values_in: cute.Tensor,
    block_root_out: cute.Tensor,
    block_root_second_out: cute.Tensor,
    block_partial_out: cute.Tensor,
    block_array_out: cute.Tensor,
    warp_partial_out: cute.Tensor,
    scoped_block_partial_out: cute.Tensor,
    scoped_block_array_out: cute.Tensor,
    scoped_warp_partial_out: cute.Tensor,
):
    _group_reduce_variant_kernel(
        values_in,
        block_root_out,
        block_root_second_out,
        block_partial_out,
        block_array_out,
        warp_partial_out,
        scoped_block_partial_out,
        scoped_block_array_out,
        scoped_warp_partial_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


def test_provider_group_reduce_cudax_root_and_direct_cub_runtime():
    cutlass.cuda.initialize_cuda_context()
    _GROUP_REDUCE_COMPAT_STORAGE.reset_uses()

    values_host = torch.arange(1, 129, dtype=torch.int32)
    values_in = values_host.cuda()
    block_root_out = torch.zeros((1,), dtype=torch.int32, device="cuda")
    block_root_second_out = torch.zeros((1,), dtype=torch.int32, device="cuda")
    block_partial_out = torch.zeros((1,), dtype=torch.int32, device="cuda")
    block_array_out = torch.zeros((1,), dtype=torch.int32, device="cuda")
    warp_partial_out = torch.zeros((2,), dtype=torch.int32, device="cuda")
    scoped_block_partial_out = torch.zeros((1,), dtype=torch.int32, device="cuda")
    scoped_block_array_out = torch.zeros((1,), dtype=torch.int32, device="cuda")
    scoped_warp_partial_out = torch.zeros((2,), dtype=torch.int32, device="cuda")

    _run_group_reduce_variants(
        from_dlpack(values_in),
        from_dlpack(block_root_out),
        from_dlpack(block_root_second_out),
        from_dlpack(block_partial_out),
        from_dlpack(block_array_out),
        from_dlpack(warp_partial_out),
        from_dlpack(scoped_block_partial_out),
        from_dlpack(scoped_block_array_out),
        from_dlpack(scoped_warp_partial_out),
    )
    torch.cuda.synchronize()

    assert block_root_out.item() == sum(range(1, 65))
    assert block_root_second_out.item() == sum(range(65, 129))
    assert block_partial_out.item() == sum(range(1, 49))
    assert block_array_out.item() == sum(range(1, 129))
    assert scoped_block_partial_out.item() == block_partial_out.item()
    assert scoped_block_array_out.item() == block_array_out.item()
    assert _GROUP_REDUCE_COMPAT_STORAGE.uses == ()
    torch.testing.assert_close(
        warp_partial_out.cpu(),
        torch.tensor([sum(range(1, 25)), sum(range(33, 57))], dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        scoped_warp_partial_out.cpu(),
        warp_partial_out.cpu(),
        atol=0,
        rtol=0,
    )
