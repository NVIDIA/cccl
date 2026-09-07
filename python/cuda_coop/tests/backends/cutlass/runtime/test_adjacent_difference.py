# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.runtime import (
    DIFF_DISC_TEMP_STORAGE as _DIFF_DISC_TEMP_STORAGE,
)
from ..support.runtime import (
    LAUNCH_CASES as _LAUNCH_CASES,
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
def _diff_discontinuity_kernel(
    values_in: cute.Tensor,
    diff_out: cute.Tensor,
    diff_right_out: cute.Tensor,
    head_out: cute.Tensor,
    tail_out: cute.Tensor,
    head2_out: cute.Tensor,
    tail2_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]
    diff = coop._block.adjacent_difference(value)
    diff_right = coop._block.adjacent_difference(
        value,
        block_adjacent_difference_type=coop._block.BlockAdjacentDifferenceType.SubtractRight,
    )
    head = coop._block.discontinuity(value)
    tail = coop._block.discontinuity(
        value,
        block_discontinuity_type=coop._block.BlockDiscontinuityType.TAILS,
    )
    head2, tail2 = coop._block.discontinuity(
        value,
        block_discontinuity_type=coop._block.BlockDiscontinuityType.HEADS_AND_TAILS,
    )

    diff_out[tidx] = diff
    diff_right_out[tidx] = diff_right
    head_out[tidx] = head
    tail_out[tidx] = tail
    head2_out[tidx] = head2
    tail2_out[tidx] = tail2


@cute.kernel
def _diff_discontinuity_temp_kernel(
    values_in: cute.Tensor,
    diff_out: cute.Tensor,
    diff_right_out: cute.Tensor,
    head_out: cute.Tensor,
    tail_out: cute.Tensor,
    head2_out: cute.Tensor,
    tail2_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]
    diff = coop._block.adjacent_difference(value, temp_storage=_DIFF_DISC_TEMP_STORAGE)
    diff_right = coop._block.adjacent_difference(
        value,
        block_adjacent_difference_type=coop._block.BlockAdjacentDifferenceType.SubtractRight,
        temp_storage=_DIFF_DISC_TEMP_STORAGE,
    )
    head = coop._block.discontinuity(value, temp_storage=_DIFF_DISC_TEMP_STORAGE)
    tail = coop._block.discontinuity(
        value,
        block_discontinuity_type=coop._block.BlockDiscontinuityType.TAILS,
        temp_storage=_DIFF_DISC_TEMP_STORAGE,
    )
    head2, tail2 = coop._block.discontinuity(
        value,
        block_discontinuity_type=coop._block.BlockDiscontinuityType.HEADS_AND_TAILS,
        temp_storage=_DIFF_DISC_TEMP_STORAGE,
    )

    diff_out[tidx] = diff
    diff_right_out[tidx] = diff_right
    head_out[tidx] = head
    tail_out[tidx] = tail
    head2_out[tidx] = head2
    tail2_out[tidx] = tail2


@cute.jit
def _run_diff_discontinuity(
    values_in: cute.Tensor,
    diff_out: cute.Tensor,
    diff_right_out: cute.Tensor,
    head_out: cute.Tensor,
    tail_out: cute.Tensor,
    head2_out: cute.Tensor,
    tail2_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _diff_discontinuity_kernel(
        values_in,
        diff_out,
        diff_right_out,
        head_out,
        tail_out,
        head2_out,
        tail2_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_diff_discontinuity_temp(
    values_in: cute.Tensor,
    diff_out: cute.Tensor,
    diff_right_out: cute.Tensor,
    head_out: cute.Tensor,
    tail_out: cute.Tensor,
    head2_out: cute.Tensor,
    tail2_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _diff_discontinuity_temp_kernel(
        values_in,
        diff_out,
        diff_right_out,
        head_out,
        tail_out,
        head2_out,
        tail2_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


def _expected_diff_heads_tails(
    values: torch.Tensor,
    *,
    flag_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, ...]:
    expected_diff = values.clone()
    expected_diff[1:] = values[1:] - values[:-1]
    expected_diff_right = values.clone()
    expected_diff_right[:-1] = values[:-1] - values[1:]
    flags_dtype = values.dtype if flag_dtype is None else flag_dtype
    expected_head = torch.zeros((int(values.numel()),), dtype=flags_dtype)
    expected_tail = torch.zeros((int(values.numel()),), dtype=flags_dtype)
    expected_head[0] = 1
    expected_tail[-1] = 1
    for idx in range(1, int(values.numel())):
        expected_head[idx] = int(values[idx] != values[idx - 1])
    for idx in range(0, int(values.numel()) - 1):
        expected_tail[idx] = int(values[idx] != values[idx + 1])
    return expected_diff, expected_diff_right, expected_head, expected_tail


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_difference_discontinuity_runtime(
    block_x: int, use_temp_storage: bool
):
    cutlass.cuda.initialize_cuda_context()
    _DIFF_DISC_TEMP_STORAGE.reset_uses()

    values_host = torch.tensor(
        [((idx // 3) + (idx % 7 == 0)) for idx in range(block_x)],
        dtype=torch.int32,
    )
    values_in = values_host.cuda()
    diff_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    diff_right_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    head_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    tail_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    head2_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    tail2_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_diff_discontinuity_temp(
            from_dlpack(values_in),
            from_dlpack(diff_out),
            from_dlpack(diff_right_out),
            from_dlpack(head_out),
            from_dlpack(tail_out),
            from_dlpack(head2_out),
            from_dlpack(tail2_out),
            block_x,
        )
    else:
        _run_diff_discontinuity(
            from_dlpack(values_in),
            from_dlpack(diff_out),
            from_dlpack(diff_right_out),
            from_dlpack(head_out),
            from_dlpack(tail_out),
            from_dlpack(head2_out),
            from_dlpack(tail2_out),
            block_x,
        )
    torch.cuda.synchronize()

    expected_diff, expected_diff_right, expected_head, expected_tail = (
        _expected_diff_heads_tails(values_host)
    )
    torch.testing.assert_close(diff_out.cpu(), expected_diff, atol=0, rtol=0)
    torch.testing.assert_close(
        diff_right_out.cpu(), expected_diff_right, atol=0, rtol=0
    )
    torch.testing.assert_close(head_out.cpu(), expected_head, atol=0, rtol=0)
    torch.testing.assert_close(tail_out.cpu(), expected_tail, atol=0, rtol=0)
    torch.testing.assert_close(head2_out.cpu(), expected_head, atol=0, rtol=0)
    torch.testing.assert_close(tail2_out.cpu(), expected_tail, atol=0, rtol=0)


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_difference_discontinuity_runtime_float32(
    block_x: int, use_temp_storage: bool
):
    cutlass.cuda.initialize_cuda_context()
    _DIFF_DISC_TEMP_STORAGE.reset_uses()

    values_host = torch.tensor(
        [float((idx // 4) + (idx % 9 == 0)) for idx in range(block_x)],
        dtype=torch.float32,
    )
    values_in = values_host.cuda()
    diff_out = torch.zeros((block_x,), dtype=torch.float32, device="cuda")
    diff_right_out = torch.zeros((block_x,), dtype=torch.float32, device="cuda")
    head_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    tail_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    head2_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    tail2_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_diff_discontinuity_temp(
            from_dlpack(values_in),
            from_dlpack(diff_out),
            from_dlpack(diff_right_out),
            from_dlpack(head_out),
            from_dlpack(tail_out),
            from_dlpack(head2_out),
            from_dlpack(tail2_out),
            block_x,
        )
    else:
        _run_diff_discontinuity(
            from_dlpack(values_in),
            from_dlpack(diff_out),
            from_dlpack(diff_right_out),
            from_dlpack(head_out),
            from_dlpack(tail_out),
            from_dlpack(head2_out),
            from_dlpack(tail2_out),
            block_x,
        )
    torch.cuda.synchronize()

    expected_diff, expected_diff_right, expected_head, expected_tail = (
        _expected_diff_heads_tails(values_host, flag_dtype=torch.int32)
    )
    torch.testing.assert_close(diff_out.cpu(), expected_diff, atol=0, rtol=0)
    torch.testing.assert_close(
        diff_right_out.cpu(), expected_diff_right, atol=0, rtol=0
    )
    torch.testing.assert_close(head_out.cpu(), expected_head, atol=0, rtol=0)
    torch.testing.assert_close(tail_out.cpu(), expected_tail, atol=0, rtol=0)
    torch.testing.assert_close(head2_out.cpu(), expected_head, atol=0, rtol=0)
    torch.testing.assert_close(tail2_out.cpu(), expected_tail, atol=0, rtol=0)


@cute.kernel
def _adjacent_difference_thread_data_kernel(
    values_in: cute.Tensor,
    predecessor_in: cute.Tensor,
    successor_in: cute.Tensor,
    scoped_full_out: cute.Tensor,
    root_full_out: cute.Tensor,
    partial_left_out: cute.Tensor,
    full_right_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    items_per_thread = cutlass.const_expr(2)
    base = tidx * items_per_thread
    items = coop.ThreadData.from_values(
        values_in[base + 0],
        values_in[base + 1],
        dtype=Int32,
    )
    group = coop.this_block()
    scoped_full = coop._block.adjacent_difference(items)
    root_full = coop.adjacent_difference(group, items)
    partial_left = coop.adjacent_difference(
        group,
        items,
        valid_items=125,
        tile_predecessor_item=predecessor_in[0],
    )
    full_right = coop.adjacent_difference(
        group,
        items,
        direction="right",
        tile_successor_item=successor_in[0],
    )
    scoped_full_out[base + 0] = scoped_full[0]
    scoped_full_out[base + 1] = scoped_full[1]
    root_full_out[base + 0] = root_full[0]
    root_full_out[base + 1] = root_full[1]
    partial_left_out[base + 0] = partial_left[0]
    partial_left_out[base + 1] = partial_left[1]
    full_right_out[base + 0] = full_right[0]
    full_right_out[base + 1] = full_right[1]


@cute.kernel
def _adjacent_difference_register_payload_kernel(
    values_in: cute.Tensor,
    predecessor_in: cute.Tensor,
    successor_in: cute.Tensor,
    scoped_full_out: cute.Tensor,
    root_full_out: cute.Tensor,
    partial_left_out: cute.Tensor,
    full_right_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    items_per_thread = cutlass.const_expr(2)
    base = tidx * items_per_thread
    items = cute.make_rmem_tensor((1, 2), Int32)
    items[0] = values_in[base + 0]
    items[1] = values_in[base + 1]
    items_ssa = items.load()
    group = coop.this_block()
    scoped_full = coop._block.adjacent_difference(items)
    root_full = coop.adjacent_difference(group, items_ssa)
    partial_left = coop.adjacent_difference(
        group,
        items_ssa,
        valid_items=125,
        tile_predecessor_item=predecessor_in[0],
    )
    full_right = coop.adjacent_difference(
        group,
        items,
        direction="right",
        tile_successor_item=successor_in[0],
    )
    scoped_full_out[base + 0] = scoped_full[0]
    scoped_full_out[base + 1] = scoped_full[1]
    root_full_out[base + 0] = root_full[0]
    root_full_out[base + 1] = root_full[1]
    partial_left_out[base + 0] = partial_left[0]
    partial_left_out[base + 1] = partial_left[1]
    full_right_out[base + 0] = full_right[0]
    full_right_out[base + 1] = full_right[1]


@cute.jit
def _run_adjacent_difference_thread_data(
    values_in: cute.Tensor,
    predecessor_in: cute.Tensor,
    successor_in: cute.Tensor,
    scoped_full_out: cute.Tensor,
    root_full_out: cute.Tensor,
    partial_left_out: cute.Tensor,
    full_right_out: cute.Tensor,
):
    _adjacent_difference_thread_data_kernel(
        values_in,
        predecessor_in,
        successor_in,
        scoped_full_out,
        root_full_out,
        partial_left_out,
        full_right_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


@cute.jit
def _run_adjacent_difference_register_payload(
    values_in: cute.Tensor,
    predecessor_in: cute.Tensor,
    successor_in: cute.Tensor,
    scoped_full_out: cute.Tensor,
    root_full_out: cute.Tensor,
    partial_left_out: cute.Tensor,
    full_right_out: cute.Tensor,
):
    _adjacent_difference_register_payload_kernel(
        values_in,
        predecessor_in,
        successor_in,
        scoped_full_out,
        root_full_out,
        partial_left_out,
        full_right_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


@pytest.mark.parametrize("payload_kind", ("thread_data", "register_payload"))
def test_provider_adjacent_difference_multi_item_payload_boundaries_and_partial_runtime(
    payload_kind: str,
):
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.tensor(
        [idx * 3 + (idx % 5) for idx in range(128)],
        dtype=torch.int32,
    )
    predecessor_host = torch.tensor([17], dtype=torch.int32)
    successor_host = torch.tensor([-9], dtype=torch.int32)
    values_in = values_host.cuda()
    predecessor_in = predecessor_host.cuda()
    successor_in = successor_host.cuda()
    scoped_full_out = torch.zeros_like(values_in)
    root_full_out = torch.zeros_like(values_in)
    partial_left_out = torch.zeros_like(values_in)
    full_right_out = torch.zeros_like(values_in)

    runner = (
        _run_adjacent_difference_register_payload
        if payload_kind == "register_payload"
        else _run_adjacent_difference_thread_data
    )
    runner(
        from_dlpack(values_in),
        from_dlpack(predecessor_in),
        from_dlpack(successor_in),
        from_dlpack(scoped_full_out),
        from_dlpack(root_full_out),
        from_dlpack(partial_left_out),
        from_dlpack(full_right_out),
    )
    torch.cuda.synchronize()

    expected_full_left = values_host.clone()
    expected_full_left[1:] = values_host[1:] - values_host[:-1]
    expected_partial_left = values_host.clone()
    expected_partial_left[0] = values_host[0] - predecessor_host[0]
    expected_partial_left[1:125] = values_host[1:125] - values_host[:124]
    expected_full_right = values_host.clone()
    expected_full_right[:-1] = values_host[:-1] - values_host[1:]
    expected_full_right[-1] = values_host[-1] - successor_host[0]

    torch.testing.assert_close(
        scoped_full_out.cpu(), expected_full_left, atol=0, rtol=0
    )
    torch.testing.assert_close(root_full_out.cpu(), expected_full_left, atol=0, rtol=0)
    torch.testing.assert_close(
        partial_left_out.cpu(), expected_partial_left, atol=0, rtol=0
    )
    torch.testing.assert_close(
        full_right_out.cpu(), expected_full_right, atol=0, rtol=0
    )
    torch.testing.assert_close(
        scoped_full_out.cpu(), root_full_out.cpu(), atol=0, rtol=0
    )


@cute.kernel
def _discontinuity_thread_data_boundary_kernel(
    values_in: cute.Tensor,
    predecessor_in: cute.Tensor,
    successor_in: cute.Tensor,
    scoped_heads_out: cute.Tensor,
    scoped_tails_out: cute.Tensor,
    root_heads_out: cute.Tensor,
    root_tails_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    items_per_thread = cutlass.const_expr(2)
    base = tidx * items_per_thread
    items = coop.ThreadData.from_values(
        values_in[base + 0],
        values_in[base + 1],
        dtype=Int32,
    )
    scoped_heads = coop._block.discontinuity_flag_heads(
        items,
        tile_predecessor_item=predecessor_in[0],
        temp_storage=_DIFF_DISC_TEMP_STORAGE,
    )
    scoped_tails = coop._block.discontinuity_flag_tails(
        items,
        tile_successor_item=successor_in[0],
        temp_storage=_DIFF_DISC_TEMP_STORAGE,
    )
    root_heads, root_tails = coop.discontinuity(
        coop.this_block(),
        items,
        mode="heads_and_tails",
        tile_predecessor_item=predecessor_in[0],
        tile_successor_item=successor_in[0],
    )
    scoped_heads_out[base + 0] = scoped_heads[0]
    scoped_heads_out[base + 1] = scoped_heads[1]
    scoped_tails_out[base + 0] = scoped_tails[0]
    scoped_tails_out[base + 1] = scoped_tails[1]
    root_heads_out[base + 0] = root_heads[0]
    root_heads_out[base + 1] = root_heads[1]
    root_tails_out[base + 0] = root_tails[0]
    root_tails_out[base + 1] = root_tails[1]


@cute.kernel
def _discontinuity_register_payload_boundary_kernel(
    values_in: cute.Tensor,
    predecessor_in: cute.Tensor,
    successor_in: cute.Tensor,
    scoped_heads_out: cute.Tensor,
    scoped_tails_out: cute.Tensor,
    root_heads_out: cute.Tensor,
    root_tails_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    items_per_thread = cutlass.const_expr(2)
    base = tidx * items_per_thread
    items = cute.make_rmem_tensor((1, 2), Int32)
    items[0] = values_in[base + 0]
    items[1] = values_in[base + 1]
    items_ssa = items.load()
    scoped_heads = coop._block.discontinuity_flag_heads(
        items_ssa,
        tile_predecessor_item=predecessor_in[0],
        temp_storage=_DIFF_DISC_TEMP_STORAGE,
    )
    scoped_tails = coop._block.discontinuity_flag_tails(
        items,
        tile_successor_item=successor_in[0],
        temp_storage=_DIFF_DISC_TEMP_STORAGE,
    )
    root_heads, root_tails = coop.discontinuity(
        coop.this_block(),
        items,
        mode="heads_and_tails",
        tile_predecessor_item=predecessor_in[0],
        tile_successor_item=successor_in[0],
    )
    scoped_heads_out[base + 0] = scoped_heads[0]
    scoped_heads_out[base + 1] = scoped_heads[1]
    scoped_tails_out[base + 0] = scoped_tails[0]
    scoped_tails_out[base + 1] = scoped_tails[1]
    root_heads_out[base + 0] = root_heads[0]
    root_heads_out[base + 1] = root_heads[1]
    root_tails_out[base + 0] = root_tails[0]
    root_tails_out[base + 1] = root_tails[1]


@cute.jit
def _run_discontinuity_thread_data_boundary(
    values_in: cute.Tensor,
    predecessor_in: cute.Tensor,
    successor_in: cute.Tensor,
    scoped_heads_out: cute.Tensor,
    scoped_tails_out: cute.Tensor,
    root_heads_out: cute.Tensor,
    root_tails_out: cute.Tensor,
):
    _discontinuity_thread_data_boundary_kernel(
        values_in,
        predecessor_in,
        successor_in,
        scoped_heads_out,
        scoped_tails_out,
        root_heads_out,
        root_tails_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


@cute.jit
def _run_discontinuity_register_payload_boundary(
    values_in: cute.Tensor,
    predecessor_in: cute.Tensor,
    successor_in: cute.Tensor,
    scoped_heads_out: cute.Tensor,
    scoped_tails_out: cute.Tensor,
    root_heads_out: cute.Tensor,
    root_tails_out: cute.Tensor,
):
    _discontinuity_register_payload_boundary_kernel(
        values_in,
        predecessor_in,
        successor_in,
        scoped_heads_out,
        scoped_tails_out,
        root_heads_out,
        root_tails_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


@pytest.mark.parametrize("payload_kind", ("thread_data", "register_payload"))
def test_provider_discontinuity_multi_item_payload_boundaries_and_root_runtime(
    payload_kind: str,
):
    cutlass.cuda.initialize_cuda_context()
    _DIFF_DISC_TEMP_STORAGE.reset_uses()

    values_host = torch.tensor(
        [idx // 3 + (idx % 11 == 0) for idx in range(128)],
        dtype=torch.int32,
    )
    predecessor_host = torch.tensor([values_host[0] - 1], dtype=torch.int32)
    successor_host = torch.tensor([values_host[-1]], dtype=torch.int32)
    values_in = values_host.cuda()
    predecessor_in = predecessor_host.cuda()
    successor_in = successor_host.cuda()
    outputs = [torch.zeros((128,), dtype=torch.int32, device="cuda") for _ in range(4)]

    runner = (
        _run_discontinuity_register_payload_boundary
        if payload_kind == "register_payload"
        else _run_discontinuity_thread_data_boundary
    )
    runner(
        from_dlpack(values_in),
        from_dlpack(predecessor_in),
        from_dlpack(successor_in),
        *(from_dlpack(output) for output in outputs),
    )
    torch.cuda.synchronize()

    expected_heads = torch.zeros((128,), dtype=torch.int32)
    expected_tails = torch.zeros((128,), dtype=torch.int32)
    expected_heads[0] = int(predecessor_host[0] != values_host[0])
    expected_heads[1:] = (values_host[:-1] != values_host[1:]).to(torch.int32)
    expected_tails[:-1] = (values_host[:-1] != values_host[1:]).to(torch.int32)
    expected_tails[-1] = int(values_host[-1] != successor_host[0])
    scoped_heads, scoped_tails, root_heads, root_tails = (
        output.cpu() for output in outputs
    )
    torch.testing.assert_close(scoped_heads, expected_heads, atol=0, rtol=0)
    torch.testing.assert_close(scoped_tails, expected_tails, atol=0, rtol=0)
    torch.testing.assert_close(root_heads, expected_heads, atol=0, rtol=0)
    torch.testing.assert_close(root_tails, expected_tails, atol=0, rtol=0)
