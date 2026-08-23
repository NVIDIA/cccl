# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.runtime import (
    LAUNCH_CASES as _LAUNCH_CASES,
)
from ..support.runtime import (
    SHUFFLE_TEMP_STORAGE as _SHUFFLE_TEMP_STORAGE,
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
def _shuffle_kernel(
    values_in: cute.Tensor,
    offset_out: cute.Tensor,
    offset_neg_out: cute.Tensor,
    rotate_out: cute.Tensor,
    root_rotate_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]
    offset_out[tidx] = coop._block.shuffle_offset(value, distance=2)
    offset_neg_out[tidx] = coop._block.shuffle_offset(value, distance=-2)
    rotate_out[tidx] = coop._block.shuffle_rotate(value, distance=67)
    root_rotate_out[tidx] = coop.shuffle(
        coop.this_block(),
        value,
        mode="rotate",
        distance=67,
    )


@cute.kernel
def _shuffle_temp_kernel(
    values_in: cute.Tensor,
    offset_out: cute.Tensor,
    offset_neg_out: cute.Tensor,
    rotate_out: cute.Tensor,
    root_rotate_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]
    offset_out[tidx] = coop._block.shuffle_offset(
        value,
        distance=2,
        temp_storage=_SHUFFLE_TEMP_STORAGE,
    )
    offset_neg_out[tidx] = coop._block.shuffle_offset(
        value,
        distance=-2,
        temp_storage=_SHUFFLE_TEMP_STORAGE,
    )
    rotate_out[tidx] = coop._block.shuffle_rotate(
        value,
        distance=67,
        temp_storage=_SHUFFLE_TEMP_STORAGE,
    )
    root_rotate_out[tidx] = coop.shuffle(
        coop.this_block(),
        value,
        mode="rotate",
        distance=67,
    )


@cute.kernel
def _shuffle_thread_data_kernel(
    values_in: cute.Tensor,
    scoped_up_out: cute.Tensor,
    root_up_out: cute.Tensor,
    down_out: cute.Tensor,
    suffix_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    items_per_thread = cutlass.const_expr(4)
    base = tidx * items_per_thread
    items = coop.ThreadData.from_values(
        values_in[base + 0],
        values_in[base + 1],
        values_in[base + 2],
        values_in[base + 3],
        dtype=Int32,
    )
    block_suffix = coop.ThreadData(1, dtype=Int32)
    block_prefix = coop.ThreadData(1, dtype=Int32)
    scoped_up = coop._block.shuffle_up(items, block_suffix=block_suffix)
    root_up = coop.shuffle(coop.this_block(), items, mode="up")
    down = coop._block.shuffle_down(items, block_prefix=block_prefix)
    if tidx < block_x:
        scoped_up_out[base + 0] = scoped_up[0]
        scoped_up_out[base + 1] = scoped_up[1]
        scoped_up_out[base + 2] = scoped_up[2]
        scoped_up_out[base + 3] = scoped_up[3]
        root_up_out[base + 0] = root_up[0]
        root_up_out[base + 1] = root_up[1]
        root_up_out[base + 2] = root_up[2]
        root_up_out[base + 3] = root_up[3]
        down_out[base + 0] = down[0]
        down_out[base + 1] = down[1]
        down_out[base + 2] = down[2]
        down_out[base + 3] = down[3]
        suffix_out[tidx] = block_suffix[0]
        prefix_out[tidx] = block_prefix[0]


@cute.kernel
def _shuffle_thread_data_temp_kernel(
    values_in: cute.Tensor,
    scoped_up_out: cute.Tensor,
    root_up_out: cute.Tensor,
    down_out: cute.Tensor,
    suffix_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    items_per_thread = cutlass.const_expr(4)
    base = tidx * items_per_thread
    items = coop.ThreadData.from_values(
        values_in[base + 0],
        values_in[base + 1],
        values_in[base + 2],
        values_in[base + 3],
        dtype=Int32,
    )
    block_suffix = coop.ThreadData(1, dtype=Int32)
    block_prefix = coop.ThreadData(1, dtype=Int32)
    scoped_up = coop._block.shuffle_up(
        items,
        block_suffix=block_suffix,
        temp_storage=_SHUFFLE_TEMP_STORAGE,
    )
    root_up = coop.shuffle(coop.this_block(), items, mode="up")
    down = coop._block.shuffle_down(
        items,
        block_prefix=block_prefix,
        temp_storage=_SHUFFLE_TEMP_STORAGE,
    )
    if tidx < block_x:
        scoped_up_out[base + 0] = scoped_up[0]
        scoped_up_out[base + 1] = scoped_up[1]
        scoped_up_out[base + 2] = scoped_up[2]
        scoped_up_out[base + 3] = scoped_up[3]
        root_up_out[base + 0] = root_up[0]
        root_up_out[base + 1] = root_up[1]
        root_up_out[base + 2] = root_up[2]
        root_up_out[base + 3] = root_up[3]
        down_out[base + 0] = down[0]
        down_out[base + 1] = down[1]
        down_out[base + 2] = down[2]
        down_out[base + 3] = down[3]
        suffix_out[tidx] = block_suffix[0]
        prefix_out[tidx] = block_prefix[0]


@cute.kernel
def _shuffle_register_payload_kernel(
    values_in: cute.Tensor,
    scoped_up_out: cute.Tensor,
    root_up_out: cute.Tensor,
    down_out: cute.Tensor,
    suffix_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    items_per_thread = cutlass.const_expr(4)
    base = tidx * items_per_thread
    items = cute.make_rmem_tensor((1, 4), Int32)
    items[0] = values_in[base + 0]
    items[1] = values_in[base + 1]
    items[2] = values_in[base + 2]
    items[3] = values_in[base + 3]
    items_ssa = items.load()
    block_suffix = coop.ThreadData(1, dtype=Int32)
    block_prefix = coop.ThreadData(1, dtype=Int32)
    scoped_up = coop._block.shuffle_up(items, block_suffix=block_suffix)
    root_up = coop.shuffle(coop.this_block(), items_ssa, mode="up")
    down = coop._block.shuffle_down(items, block_prefix=block_prefix)
    if tidx < block_x:
        scoped_up_out[base + 0] = scoped_up[0]
        scoped_up_out[base + 1] = scoped_up[1]
        scoped_up_out[base + 2] = scoped_up[2]
        scoped_up_out[base + 3] = scoped_up[3]
        root_up_out[base + 0] = root_up[0]
        root_up_out[base + 1] = root_up[1]
        root_up_out[base + 2] = root_up[2]
        root_up_out[base + 3] = root_up[3]
        down_out[base + 0] = down[0]
        down_out[base + 1] = down[1]
        down_out[base + 2] = down[2]
        down_out[base + 3] = down[3]
        suffix_out[tidx] = block_suffix[0]
        prefix_out[tidx] = block_prefix[0]


@cute.kernel
def _shuffle_register_payload_temp_kernel(
    values_in: cute.Tensor,
    scoped_up_out: cute.Tensor,
    root_up_out: cute.Tensor,
    down_out: cute.Tensor,
    suffix_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    items_per_thread = cutlass.const_expr(4)
    base = tidx * items_per_thread
    items = cute.make_rmem_tensor((1, 4), Int32)
    items[0] = values_in[base + 0]
    items[1] = values_in[base + 1]
    items[2] = values_in[base + 2]
    items[3] = values_in[base + 3]
    items_ssa = items.load()
    block_suffix = coop.ThreadData(1, dtype=Int32)
    block_prefix = coop.ThreadData(1, dtype=Int32)
    scoped_up = coop._block.shuffle_up(
        items,
        block_suffix=block_suffix,
        temp_storage=_SHUFFLE_TEMP_STORAGE,
    )
    root_up = coop.shuffle(coop.this_block(), items_ssa, mode="up")
    down = coop._block.shuffle_down(
        items,
        block_prefix=block_prefix,
        temp_storage=_SHUFFLE_TEMP_STORAGE,
    )
    if tidx < block_x:
        scoped_up_out[base + 0] = scoped_up[0]
        scoped_up_out[base + 1] = scoped_up[1]
        scoped_up_out[base + 2] = scoped_up[2]
        scoped_up_out[base + 3] = scoped_up[3]
        root_up_out[base + 0] = root_up[0]
        root_up_out[base + 1] = root_up[1]
        root_up_out[base + 2] = root_up[2]
        root_up_out[base + 3] = root_up[3]
        down_out[base + 0] = down[0]
        down_out[base + 1] = down[1]
        down_out[base + 2] = down[2]
        down_out[base + 3] = down[3]
        suffix_out[tidx] = block_suffix[0]
        prefix_out[tidx] = block_prefix[0]


@cute.jit
def _run_shuffle(
    values_in: cute.Tensor,
    offset_out: cute.Tensor,
    offset_neg_out: cute.Tensor,
    rotate_out: cute.Tensor,
    root_rotate_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _shuffle_kernel(
        values_in,
        offset_out,
        offset_neg_out,
        rotate_out,
        root_rotate_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_shuffle_temp(
    values_in: cute.Tensor,
    offset_out: cute.Tensor,
    offset_neg_out: cute.Tensor,
    rotate_out: cute.Tensor,
    root_rotate_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _shuffle_temp_kernel(
        values_in,
        offset_out,
        offset_neg_out,
        rotate_out,
        root_rotate_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_shuffle_thread_data(
    values_in: cute.Tensor,
    scoped_up_out: cute.Tensor,
    root_up_out: cute.Tensor,
    down_out: cute.Tensor,
    suffix_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _shuffle_thread_data_kernel(
        values_in,
        scoped_up_out,
        root_up_out,
        down_out,
        suffix_out,
        prefix_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_shuffle_thread_data_temp(
    values_in: cute.Tensor,
    scoped_up_out: cute.Tensor,
    root_up_out: cute.Tensor,
    down_out: cute.Tensor,
    suffix_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _shuffle_thread_data_temp_kernel(
        values_in,
        scoped_up_out,
        root_up_out,
        down_out,
        suffix_out,
        prefix_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_shuffle_register_payload(
    values_in: cute.Tensor,
    scoped_up_out: cute.Tensor,
    root_up_out: cute.Tensor,
    down_out: cute.Tensor,
    suffix_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _shuffle_register_payload_kernel(
        values_in,
        scoped_up_out,
        root_up_out,
        down_out,
        suffix_out,
        prefix_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_shuffle_register_payload_temp(
    values_in: cute.Tensor,
    scoped_up_out: cute.Tensor,
    root_up_out: cute.Tensor,
    down_out: cute.Tensor,
    suffix_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _shuffle_register_payload_temp_kernel(
        values_in,
        scoped_up_out,
        root_up_out,
        down_out,
        suffix_out,
        prefix_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_shuffle_runtime(block_x: int, use_temp_storage: bool):
    cutlass.cuda.initialize_cuda_context()
    _SHUFFLE_TEMP_STORAGE.reset_uses()

    values_host = torch.arange(block_x, dtype=torch.int32)
    values_in = values_host.cuda()
    offset_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    offset_neg_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    rotate_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    root_rotate_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    runner = _run_shuffle_temp if use_temp_storage else _run_shuffle
    runner(
        from_dlpack(values_in),
        from_dlpack(offset_out),
        from_dlpack(offset_neg_out),
        from_dlpack(rotate_out),
        from_dlpack(root_rotate_out),
        block_x,
    )
    torch.cuda.synchronize()

    expected_positive = values_host.clone()
    expected_positive[:-2] = values_host[2:]
    expected_negative = values_host.clone()
    expected_negative[2:] = values_host[:-2]
    expected_rotate = torch.roll(values_host, shifts=-3)

    torch.testing.assert_close(offset_out.cpu(), expected_positive, atol=0, rtol=0)
    torch.testing.assert_close(offset_neg_out.cpu(), expected_negative, atol=0, rtol=0)
    torch.testing.assert_close(rotate_out.cpu(), expected_rotate, atol=0, rtol=0)
    torch.testing.assert_close(root_rotate_out.cpu(), expected_rotate, atol=0, rtol=0)


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
@pytest.mark.parametrize("payload_kind", ("thread_data", "register_payload"))
def test_provider_shuffle_runtime_multi_item_payloads_flattened(
    block_x: int,
    use_temp_storage: bool,
    payload_kind: str,
):
    cutlass.cuda.initialize_cuda_context()
    _SHUFFLE_TEMP_STORAGE.reset_uses()

    items_per_thread = 4
    total_items = block_x * items_per_thread
    values_host = torch.arange(total_items, dtype=torch.int32)
    values_in = values_host.cuda()
    scoped_up_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    root_up_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    down_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    suffix_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    prefix_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    if payload_kind == "register_payload":
        runner = (
            _run_shuffle_register_payload_temp
            if use_temp_storage
            else _run_shuffle_register_payload
        )
    else:
        runner = (
            _run_shuffle_thread_data_temp
            if use_temp_storage
            else _run_shuffle_thread_data
        )
    runner(
        from_dlpack(values_in),
        from_dlpack(scoped_up_out),
        from_dlpack(root_up_out),
        from_dlpack(down_out),
        from_dlpack(suffix_out),
        from_dlpack(prefix_out),
        block_x,
    )
    torch.cuda.synchronize()

    expected_up = values_host.clone()
    expected_up[1:] = values_host[:-1]
    expected_down = values_host.clone()
    expected_down[:-1] = values_host[1:]
    expected_suffix = torch.full(
        (block_x,), int(values_host[-1].item()), dtype=torch.int32
    )
    expected_prefix = torch.full(
        (block_x,), int(values_host[0].item()), dtype=torch.int32
    )

    torch.testing.assert_close(scoped_up_out.cpu(), expected_up, atol=0, rtol=0)
    torch.testing.assert_close(root_up_out.cpu(), expected_up, atol=0, rtol=0)
    torch.testing.assert_close(down_out.cpu(), expected_down, atol=0, rtol=0)
    torch.testing.assert_close(suffix_out.cpu(), expected_suffix, atol=0, rtol=0)
    torch.testing.assert_close(prefix_out.cpu(), expected_prefix, atol=0, rtol=0)
