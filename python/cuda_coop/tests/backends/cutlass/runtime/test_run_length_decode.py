# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.runtime import (
    LAUNCH_CASES as _LAUNCH_CASES,
)
from ..support.runtime import (
    RUN_LENGTH_TEMP_STORAGE as _RUN_LENGTH_TEMP_STORAGE,
)
from ..support.runtime import (
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
def _run_length_decode_kernel(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    decoded_out: cute.Tensor,
    offsets_out: cute.Tensor,
    total_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    runs_per_thread = cutlass.const_expr(2)
    decoded_items_per_thread = cutlass.const_expr(4)
    run_base = tidx * runs_per_thread
    runs = coop.ThreadData.from_values(
        values_in[run_base + 0],
        values_in[run_base + 1],
        dtype=Uint32,
    )
    lengths = coop.ThreadData.from_values(
        lengths_in[run_base + 0],
        lengths_in[run_base + 1],
        dtype=Uint32,
    )
    relative_offsets = coop.ThreadData(decoded_items_per_thread, dtype=Uint32)
    total_decoded_size = coop.ThreadData(1, dtype=Uint32)
    run_length = coop._block.run_length(
        runs,
        lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        total_decoded_size=total_decoded_size,
    )
    decoded = run_length.decode(relative_offsets=relative_offsets)
    out_base = tidx * decoded_items_per_thread
    if tidx < block_x:
        decoded_out[out_base + 0] = decoded[0]
        decoded_out[out_base + 1] = decoded[1]
        decoded_out[out_base + 2] = decoded[2]
        decoded_out[out_base + 3] = decoded[3]
        offsets_out[out_base + 0] = relative_offsets[0]
        offsets_out[out_base + 1] = relative_offsets[1]
        offsets_out[out_base + 2] = relative_offsets[2]
        offsets_out[out_base + 3] = relative_offsets[3]
        total_out[tidx] = total_decoded_size[0]


@cute.kernel
def _run_length_decode_temp_kernel(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    decoded_out: cute.Tensor,
    offsets_out: cute.Tensor,
    total_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    runs_per_thread = cutlass.const_expr(2)
    decoded_items_per_thread = cutlass.const_expr(4)
    run_base = tidx * runs_per_thread
    runs = coop.ThreadData.from_values(
        values_in[run_base + 0],
        values_in[run_base + 1],
        dtype=Uint32,
    )
    lengths = coop.ThreadData.from_values(
        lengths_in[run_base + 0],
        lengths_in[run_base + 1],
        dtype=Uint32,
    )
    relative_offsets = coop.ThreadData(decoded_items_per_thread, dtype=Uint32)
    total_decoded_size = coop.ThreadData(1, dtype=Uint32)
    decoded = coop._block.run_length_decode(
        runs,
        lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        relative_offsets=relative_offsets,
        total_decoded_size=total_decoded_size,
        temp_storage=_RUN_LENGTH_TEMP_STORAGE,
    )
    out_base = tidx * decoded_items_per_thread
    if tidx < block_x:
        decoded_out[out_base + 0] = decoded[0]
        decoded_out[out_base + 1] = decoded[1]
        decoded_out[out_base + 2] = decoded[2]
        decoded_out[out_base + 3] = decoded[3]
        offsets_out[out_base + 0] = relative_offsets[0]
        offsets_out[out_base + 1] = relative_offsets[1]
        offsets_out[out_base + 2] = relative_offsets[2]
        offsets_out[out_base + 3] = relative_offsets[3]
        total_out[tidx] = total_decoded_size[0]


@cute.jit
def _run_run_length_decode(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    decoded_out: cute.Tensor,
    offsets_out: cute.Tensor,
    total_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _run_length_decode_kernel(
        values_in,
        lengths_in,
        decoded_out,
        offsets_out,
        total_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_run_length_decode_temp(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    decoded_out: cute.Tensor,
    offsets_out: cute.Tensor,
    total_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _run_length_decode_temp_kernel(
        values_in,
        lengths_in,
        decoded_out,
        offsets_out,
        total_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.kernel
def _run_length_decode_register_payload_root_scoped_parity_kernel(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    root_out: cute.Tensor,
    scoped_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    runs_per_thread = cutlass.const_expr(2)
    decoded_items_per_thread = cutlass.const_expr(4)
    run_base = tidx * runs_per_thread
    runs = cute.make_rmem_tensor((1, 2), Uint32)
    runs[0] = values_in[run_base + 0]
    runs[1] = values_in[run_base + 1]
    lengths = cute.make_rmem_tensor((1, 2), Uint32)
    lengths[0] = lengths_in[run_base + 0]
    lengths[1] = lengths_in[run_base + 1]
    root = coop.run_length_decode(
        coop.this_block(),
        runs,
        lengths.load(),
        decoded_items_per_thread=decoded_items_per_thread,
    )
    scoped = coop._block.run_length_decode(
        runs.load(),
        lengths,
        decoded_items_per_thread=decoded_items_per_thread,
    )
    out_base = tidx * decoded_items_per_thread
    root_out[out_base + 0] = root[0]
    root_out[out_base + 1] = root[1]
    root_out[out_base + 2] = root[2]
    root_out[out_base + 3] = root[3]
    scoped_out[out_base + 0] = scoped[0]
    scoped_out[out_base + 1] = scoped[1]
    scoped_out[out_base + 2] = scoped[2]
    scoped_out[out_base + 3] = scoped[3]


@cute.kernel
def _run_length_decode_thread_data_root_scoped_parity_kernel(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    root_out: cute.Tensor,
    scoped_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    runs_per_thread = cutlass.const_expr(2)
    decoded_items_per_thread = cutlass.const_expr(4)
    run_base = tidx * runs_per_thread
    runs = coop.ThreadData.from_values(
        values_in[run_base + 0],
        values_in[run_base + 1],
        dtype=Uint32,
    )
    lengths = coop.ThreadData.from_values(
        lengths_in[run_base + 0],
        lengths_in[run_base + 1],
        dtype=Uint32,
    )
    root = coop.run_length_decode(
        coop.this_block(),
        runs,
        lengths,
        decoded_items_per_thread=decoded_items_per_thread,
    )
    scoped = coop._block.run_length_decode(
        runs,
        lengths,
        decoded_items_per_thread=decoded_items_per_thread,
    )
    out_base = tidx * decoded_items_per_thread
    root_out[out_base + 0] = root[0]
    root_out[out_base + 1] = root[1]
    root_out[out_base + 2] = root[2]
    root_out[out_base + 3] = root[3]
    scoped_out[out_base + 0] = scoped[0]
    scoped_out[out_base + 1] = scoped[1]
    scoped_out[out_base + 2] = scoped[2]
    scoped_out[out_base + 3] = scoped[3]


@cute.jit
def _run_run_length_decode_register_payload_root_scoped_parity(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    root_out: cute.Tensor,
    scoped_out: cute.Tensor,
):
    _run_length_decode_register_payload_root_scoped_parity_kernel(
        values_in,
        lengths_in,
        root_out,
        scoped_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


@cute.jit
def _run_run_length_decode_thread_data_root_scoped_parity(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    root_out: cute.Tensor,
    scoped_out: cute.Tensor,
):
    _run_length_decode_thread_data_root_scoped_parity_kernel(
        values_in,
        lengths_in,
        root_out,
        scoped_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


@cute.kernel
def _run_length_decode_scalar_root_scoped_kernel(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    root_out: cute.Tensor,
    scoped_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    decoded_items_per_thread = cutlass.const_expr(2)
    root = coop.run_length_decode(
        coop.this_block(),
        values_in[tidx],
        lengths_in[tidx],
        decoded_items_per_thread=decoded_items_per_thread,
    )
    scoped = coop._block.run_length_decode(
        values_in[tidx],
        lengths_in[tidx],
        decoded_items_per_thread=decoded_items_per_thread,
    )
    out_base = tidx * decoded_items_per_thread
    root_out[out_base + 0] = root[0]
    root_out[out_base + 1] = root[1]
    scoped_out[out_base + 0] = scoped[0]
    scoped_out[out_base + 1] = scoped[1]


@cute.jit
def _run_run_length_decode_scalar_root_scoped(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    root_out: cute.Tensor,
    scoped_out: cute.Tensor,
):
    _run_length_decode_scalar_root_scoped_kernel(
        values_in,
        lengths_in,
        root_out,
        scoped_out,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1))


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_run_length_decode_runtime(block_x: int, use_temp_storage: bool):
    cutlass.cuda.initialize_cuda_context()
    _RUN_LENGTH_TEMP_STORAGE.reset_uses()

    runs_per_thread = 2
    decoded_items_per_thread = 4
    total_runs = block_x * runs_per_thread
    window_size = block_x * decoded_items_per_thread
    values_host = torch.tensor(
        [idx for idx in range(total_runs)],
        dtype=torch.uint32,
    )
    lengths_host = torch.full((total_runs,), 2, dtype=torch.uint32)
    values_in = values_host.cuda()
    lengths_in = lengths_host.cuda()
    decoded_out = torch.zeros((window_size,), dtype=torch.uint32, device="cuda")
    offsets_out = torch.zeros((window_size,), dtype=torch.uint32, device="cuda")
    total_out = torch.zeros((block_x,), dtype=torch.uint32, device="cuda")

    if use_temp_storage:
        _run_run_length_decode_temp(
            from_dlpack(values_in),
            from_dlpack(lengths_in),
            from_dlpack(decoded_out),
            from_dlpack(offsets_out),
            from_dlpack(total_out),
            block_x,
        )
    else:
        _run_run_length_decode(
            from_dlpack(values_in),
            from_dlpack(lengths_in),
            from_dlpack(decoded_out),
            from_dlpack(offsets_out),
            from_dlpack(total_out),
            block_x,
        )
    torch.cuda.synchronize()

    expected_decoded = torch.tensor(
        [idx // 2 for idx in range(window_size)],
        dtype=torch.uint32,
    )
    expected_offsets = torch.tensor(
        [idx % 2 for idx in range(window_size)],
        dtype=torch.uint32,
    )
    expected_total = torch.full((block_x,), window_size, dtype=torch.uint32)
    torch.testing.assert_close(decoded_out.cpu(), expected_decoded, atol=0, rtol=0)
    torch.testing.assert_close(offsets_out.cpu(), expected_offsets, atol=0, rtol=0)
    torch.testing.assert_close(total_out.cpu(), expected_total, atol=0, rtol=0)


@pytest.mark.parametrize("use_register_payload", [False, True])
def test_provider_run_length_decode_root_scoped_runtime_parity(
    use_register_payload: bool,
):
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.arange(128, dtype=torch.int64).to(torch.uint32)
    lengths_host = torch.full((128,), 2, dtype=torch.uint32)
    root_out = torch.zeros((256,), dtype=torch.uint32, device="cuda")
    scoped_out = torch.zeros((256,), dtype=torch.uint32, device="cuda")

    runner = (
        _run_run_length_decode_register_payload_root_scoped_parity
        if use_register_payload
        else _run_run_length_decode_thread_data_root_scoped_parity
    )
    runner(
        from_dlpack(values_host.cuda()),
        from_dlpack(lengths_host.cuda()),
        from_dlpack(root_out),
        from_dlpack(scoped_out),
    )
    torch.cuda.synchronize()

    expected = torch.arange(256, dtype=torch.int64).div(2, rounding_mode="floor")
    expected = expected.to(torch.uint32)
    torch.testing.assert_close(root_out, scoped_out, atol=0, rtol=0)
    torch.testing.assert_close(root_out.cpu(), expected, atol=0, rtol=0)


def test_provider_run_length_decode_scalar_root_scoped_runtime_parity():
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.arange(32, dtype=torch.int64).to(torch.uint32)
    lengths_host = torch.full((32,), 2, dtype=torch.uint32)
    root_out = torch.zeros((64,), dtype=torch.uint32, device="cuda")
    scoped_out = torch.zeros((64,), dtype=torch.uint32, device="cuda")

    _run_run_length_decode_scalar_root_scoped(
        from_dlpack(values_host.cuda()),
        from_dlpack(lengths_host.cuda()),
        from_dlpack(root_out),
        from_dlpack(scoped_out),
    )
    torch.cuda.synchronize()

    expected = torch.arange(64, dtype=torch.int64).div(2, rounding_mode="floor")
    expected = expected.to(torch.uint32)
    torch.testing.assert_close(root_out, scoped_out, atol=0, rtol=0)
    torch.testing.assert_close(root_out.cpu(), expected, atol=0, rtol=0)


@cute.kernel
def _run_length_decode_window_kernel(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    decoded_out: cute.Tensor,
    offsets_out: cute.Tensor,
    total_out: cute.Tensor,
    decoded_window_offset: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    runs_per_thread = cutlass.const_expr(2)
    decoded_items_per_thread = cutlass.const_expr(4)
    run_base = tidx * runs_per_thread
    runs = coop.ThreadData.from_values(
        values_in[run_base + 0],
        values_in[run_base + 1],
        dtype=Uint32,
    )
    lengths = coop.ThreadData.from_values(
        lengths_in[run_base + 0],
        lengths_in[run_base + 1],
        dtype=Uint32,
    )
    relative_offsets = coop.ThreadData(decoded_items_per_thread, dtype=Uint32)
    total_decoded_size = coop.ThreadData(1, dtype=Uint32)
    decoded = coop._block.run_length_decode(
        runs,
        lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        decoded_window_offset=decoded_window_offset,
        relative_offsets=relative_offsets,
        total_decoded_size=total_decoded_size,
    )
    out_base = tidx * decoded_items_per_thread
    decoded_out[out_base + 0] = decoded[0]
    decoded_out[out_base + 1] = decoded[1]
    decoded_out[out_base + 2] = decoded[2]
    decoded_out[out_base + 3] = decoded[3]
    offsets_out[out_base + 0] = relative_offsets[0]
    offsets_out[out_base + 1] = relative_offsets[1]
    offsets_out[out_base + 2] = relative_offsets[2]
    offsets_out[out_base + 3] = relative_offsets[3]
    total_out[tidx] = total_decoded_size[0]


@cute.kernel
def _run_length_decode_window_temp_kernel(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    decoded_out: cute.Tensor,
    offsets_out: cute.Tensor,
    total_out: cute.Tensor,
    decoded_window_offset: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    runs_per_thread = cutlass.const_expr(2)
    decoded_items_per_thread = cutlass.const_expr(4)
    run_base = tidx * runs_per_thread
    runs = coop.ThreadData.from_values(
        values_in[run_base + 0],
        values_in[run_base + 1],
        dtype=Uint32,
    )
    lengths = coop.ThreadData.from_values(
        lengths_in[run_base + 0],
        lengths_in[run_base + 1],
        dtype=Uint32,
    )
    relative_offsets = coop.ThreadData(decoded_items_per_thread, dtype=Uint32)
    total_decoded_size = coop.ThreadData(1, dtype=Uint32)
    decoded = coop._block.run_length_decode(
        runs,
        lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        decoded_window_offset=decoded_window_offset,
        relative_offsets=relative_offsets,
        total_decoded_size=total_decoded_size,
        temp_storage=_RUN_LENGTH_TEMP_STORAGE,
    )
    out_base = tidx * decoded_items_per_thread
    decoded_out[out_base + 0] = decoded[0]
    decoded_out[out_base + 1] = decoded[1]
    decoded_out[out_base + 2] = decoded[2]
    decoded_out[out_base + 3] = decoded[3]
    offsets_out[out_base + 0] = relative_offsets[0]
    offsets_out[out_base + 1] = relative_offsets[1]
    offsets_out[out_base + 2] = relative_offsets[2]
    offsets_out[out_base + 3] = relative_offsets[3]
    total_out[tidx] = total_decoded_size[0]


@cute.jit
def _run_run_length_decode_window(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    decoded_out: cute.Tensor,
    offsets_out: cute.Tensor,
    total_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    decoded_window_offset: cutlass.Constexpr,
):
    _run_length_decode_window_kernel(
        values_in,
        lengths_in,
        decoded_out,
        offsets_out,
        total_out,
        decoded_window_offset,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_run_length_decode_window_temp(
    values_in: cute.Tensor,
    lengths_in: cute.Tensor,
    decoded_out: cute.Tensor,
    offsets_out: cute.Tensor,
    total_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    decoded_window_offset: cutlass.Constexpr,
):
    _run_length_decode_window_temp_kernel(
        values_in,
        lengths_in,
        decoded_out,
        offsets_out,
        total_out,
        decoded_window_offset,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@pytest.mark.parametrize(
    "block_x,use_temp_storage,decoded_window_offset",
    [(32, False, 5), (64, True, 7)],
)
def test_provider_run_length_decode_runtime_window_offset(
    block_x: int,
    use_temp_storage: bool,
    decoded_window_offset: int,
):
    cutlass.cuda.initialize_cuda_context()
    _RUN_LENGTH_TEMP_STORAGE.reset_uses()

    runs_per_thread = 2
    decoded_items_per_thread = 4
    total_runs = block_x * runs_per_thread
    window_size = block_x * decoded_items_per_thread
    values_host = (torch.arange(total_runs, dtype=torch.int64) + 100).to(torch.uint32)
    lengths_i64 = (torch.arange(total_runs, dtype=torch.int64) % 3) + 1
    required_decoded = decoded_window_offset + window_size
    decoded_total = int(lengths_i64.sum().item())
    if decoded_total < required_decoded:
        lengths_i64[-1] += required_decoded - decoded_total
        decoded_total = required_decoded
    lengths_host = lengths_i64.to(torch.uint32)
    values_in = values_host.cuda()
    lengths_in = lengths_host.cuda()
    decoded_out = torch.zeros((window_size,), dtype=torch.uint32, device="cuda")
    offsets_out = torch.zeros((window_size,), dtype=torch.uint32, device="cuda")
    total_out = torch.zeros((block_x,), dtype=torch.uint32, device="cuda")

    if use_temp_storage:
        _run_run_length_decode_window_temp(
            from_dlpack(values_in),
            from_dlpack(lengths_in),
            from_dlpack(decoded_out),
            from_dlpack(offsets_out),
            from_dlpack(total_out),
            block_x,
            decoded_window_offset,
        )
    else:
        _run_run_length_decode_window(
            from_dlpack(values_in),
            from_dlpack(lengths_in),
            from_dlpack(decoded_out),
            from_dlpack(offsets_out),
            from_dlpack(total_out),
            block_x,
            decoded_window_offset,
        )
    torch.cuda.synchronize()

    decoded_values = []
    relative_offsets = []
    for value, length in zip(values_host.tolist(), lengths_host.tolist()):
        for offset in range(length):
            decoded_values.append(value)
            relative_offsets.append(offset)
    window = slice(decoded_window_offset, decoded_window_offset + window_size)
    expected_decoded = torch.tensor(decoded_values[window], dtype=torch.uint32)
    expected_offsets = torch.tensor(relative_offsets[window], dtype=torch.uint32)
    expected_total = torch.full((block_x,), decoded_total, dtype=torch.uint32)

    torch.testing.assert_close(decoded_out.cpu(), expected_decoded, atol=0, rtol=0)
    torch.testing.assert_close(offsets_out.cpu(), expected_offsets, atol=0, rtol=0)
    torch.testing.assert_close(total_out.cpu(), expected_total, atol=0, rtol=0)


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    [(32, False), (64, True)],
)
def test_provider_run_length_decode_runtime_oob_sentinel(
    block_x: int,
    use_temp_storage: bool,
):
    cutlass.cuda.initialize_cuda_context()
    _RUN_LENGTH_TEMP_STORAGE.reset_uses()

    runs_per_thread = 2
    decoded_items_per_thread = 4
    total_runs = block_x * runs_per_thread
    window_size = block_x * decoded_items_per_thread
    decoded_window_offset = total_runs - 2
    values_host = (torch.arange(total_runs, dtype=torch.int64) + 100).to(torch.uint32)
    lengths_host = torch.ones((total_runs,), dtype=torch.uint32)
    values_in = values_host.cuda()
    lengths_in = lengths_host.cuda()
    decoded_out = torch.zeros((window_size,), dtype=torch.uint32, device="cuda")
    offsets_out = torch.zeros((window_size,), dtype=torch.uint32, device="cuda")
    total_out = torch.zeros((block_x,), dtype=torch.uint32, device="cuda")

    if use_temp_storage:
        _run_run_length_decode_window_temp(
            from_dlpack(values_in),
            from_dlpack(lengths_in),
            from_dlpack(decoded_out),
            from_dlpack(offsets_out),
            from_dlpack(total_out),
            block_x,
            decoded_window_offset,
        )
    else:
        _run_run_length_decode_window(
            from_dlpack(values_in),
            from_dlpack(lengths_in),
            from_dlpack(decoded_out),
            from_dlpack(offsets_out),
            from_dlpack(total_out),
            block_x,
            decoded_window_offset,
        )
    torch.cuda.synchronize()

    expected_decoded = torch.zeros((window_size,), dtype=torch.uint32)
    expected_offsets = torch.full(
        (window_size,),
        torch.iinfo(torch.uint32).max,
        dtype=torch.uint32,
    )
    valid_items = total_runs - decoded_window_offset
    expected_decoded[:valid_items] = values_host[decoded_window_offset:]
    expected_offsets[:valid_items] = 0
    expected_total = torch.full((block_x,), total_runs, dtype=torch.uint32)

    torch.testing.assert_close(decoded_out.cpu(), expected_decoded, atol=0, rtol=0)
    torch.testing.assert_close(offsets_out.cpu(), expected_offsets, atol=0, rtol=0)
    torch.testing.assert_close(total_out.cpu(), expected_total, atol=0, rtol=0)
