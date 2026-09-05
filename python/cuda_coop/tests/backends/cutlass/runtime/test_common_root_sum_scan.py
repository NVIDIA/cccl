# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

import pytest

import cuda.coop.cutlass as cutlass_coop
from cuda import coop

from ..support.runtime import (
    Int32,
    cute,
    cutlass,
    from_dlpack,
    runtime_pytestmark,
    torch,
)

pytestmark = [
    *runtime_pytestmark,
    pytest.mark.usefixtures("qualified_cutlass_backend"),
]

_BLOCK_THREADS = 64
_ITEMS_PER_THREAD = 2
_ITEM_COUNT = _BLOCK_THREADS * _ITEMS_PER_THREAD
_SEGMENT_COUNT = 20
_SENTINEL = -999
_INT32_LOWEST = -2_147_483_648


def _store_items(output: cute.Tensor, segment: int, items) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    offset = segment * _ITEM_COUNT + tidx * _ITEMS_PER_THREAD
    output[offset] = items[0]
    output[offset + 1] = items[1]


def _store_scalar(output: cute.Tensor, segment: int, value) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    output[segment * _ITEM_COUNT + tidx] = value


def _sum_scan_body(
    api: Any,
    values: cute.Tensor,
    output: cute.Tensor,
) -> tuple[Any, Any, Any]:
    tidx, _, _ = cute.arch.thread_idx()
    block = api.this_block()
    storage = api.TempStorage()
    items = api.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
    items[0] = values[tidx * _ITEMS_PER_THREAD]
    items[1] = values[tidx * _ITEMS_PER_THREAD + 1]

    default_scan = api.scan(block, items, temp_storage=storage)
    inclusive_mode_scan = api.scan(
        block,
        items,
        mode="inclusive",
        temp_storage=storage,
    )
    exclusive_sum = api.exclusive_sum(block, items, temp_storage=storage)
    inclusive_sum = api.inclusive_sum(block, items, temp_storage=storage)
    exclusive_max = api.exclusive_scan(
        block,
        items,
        scan_op="max",
        initial_value=_INT32_LOWEST,
        temp_storage=storage,
    )
    inclusive_max = api.inclusive_scan(
        block,
        items,
        scan_op="max",
        temp_storage=storage,
    )
    block_sum = api.sum(block, items)
    partial_block_sum = api.sum(
        block,
        values[tidx],
        broadcast=False,
        valid_items=47,
        algorithm="raking",
    )
    block_max = api.reduce(
        block,
        values[tidx],
        binary_op="max",
        broadcast=False,
        algorithm="raking",
    )

    # Observe the source only after every transforming operation. This directly
    # enforces the common V1 non-mutation rule.
    _store_items(output, 0, items)
    _store_items(output, 1, default_scan)
    _store_items(output, 2, inclusive_mode_scan)
    _store_items(output, 3, exclusive_sum)
    _store_items(output, 4, inclusive_sum)
    _store_items(output, 5, exclusive_max)
    _store_items(output, 6, inclusive_max)
    _store_scalar(output, 7, block_sum)
    value = values[tidx]
    warp = api.this_warp()
    _store_scalar(output, 9, api.scan(warp, value))
    _store_scalar(output, 10, api.scan(warp, value, mode="inclusive"))
    _store_scalar(output, 11, api.exclusive_sum(warp, value))
    _store_scalar(output, 12, api.inclusive_sum(warp, value))
    _store_scalar(
        output,
        13,
        api.exclusive_scan(
            warp,
            value,
            scan_op="max",
            initial_value=_INT32_LOWEST,
        ),
    )
    _store_scalar(output, 14, api.inclusive_scan(warp, value, scan_op="max"))
    _store_scalar(output, 15, api.sum(warp, value))
    _store_scalar(output, 16, api.sum(warp.group_by(8), value))
    _store_scalar(output, 17, api.sum(api.this_thread(), value))
    warp_max = api.reduce(
        warp,
        value,
        binary_op="max",
        broadcast=False,
        valid_items=24,
    )
    return partial_block_sum, block_max, warp_max


@cute.kernel
def _common_sum_scan_kernel(values: cute.Tensor, output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    partial_block_sum, block_max, warp_max = _sum_scan_body(coop, values, output)
    if tidx == 0:
        output[8 * _ITEM_COUNT] = partial_block_sum
        output[18 * _ITEM_COUNT] = block_max
    if tidx % 32 == 0:
        output[19 * _ITEM_COUNT + tidx] = warp_max


@cute.kernel
def _qualified_sum_scan_kernel(values: cute.Tensor, output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    partial_block_sum, block_max, warp_max = _sum_scan_body(
        cutlass_coop,
        values,
        output,
    )
    if tidx == 0:
        output[8 * _ITEM_COUNT] = partial_block_sum
        output[18 * _ITEM_COUNT] = block_max
    if tidx % 32 == 0:
        output[19 * _ITEM_COUNT + tidx] = warp_max


@cute.jit
def _run_common_sum_scan(values: cute.Tensor, output: cute.Tensor):
    _common_sum_scan_kernel(values, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


@cute.jit
def _run_qualified_sum_scan(values: cute.Tensor, output: cute.Tensor):
    _qualified_sum_scan_kernel(values, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


def _expected_output(values: torch.Tensor) -> torch.Tensor:
    expected = torch.full(
        (_SEGMENT_COUNT, _ITEM_COUNT),
        _SENTINEL,
        dtype=torch.int32,
    )
    inclusive = torch.cumsum(values.to(torch.int64), dim=0).to(torch.int32)
    exclusive = inclusive - values
    inclusive_max = torch.cummax(values, dim=0).values
    exclusive_max = torch.cat(
        (torch.tensor([_INT32_LOWEST], dtype=torch.int32), inclusive_max[:-1])
    )

    expected[0] = values
    expected[1] = exclusive
    expected[2] = inclusive
    expected[3] = exclusive
    expected[4] = inclusive
    expected[5] = exclusive_max
    expected[6] = inclusive_max
    expected[7, :_BLOCK_THREADS] = int(values.sum())
    expected[8, 0] = int(values[:47].sum())

    warp_values = values[:_BLOCK_THREADS].reshape(2, 32)
    warp_inclusive = torch.cumsum(warp_values.to(torch.int64), dim=1).to(torch.int32)
    warp_exclusive = warp_inclusive - warp_values
    warp_inclusive_max = torch.cummax(warp_values, dim=1).values
    warp_exclusive_max = torch.cat(
        (
            torch.full((2, 1), _INT32_LOWEST, dtype=torch.int32),
            warp_inclusive_max[:, :-1],
        ),
        dim=1,
    )
    warp_totals = warp_values.sum(dim=1).to(torch.int32).repeat_interleave(32)
    subgroup_totals = (
        warp_values.reshape(8, 8).sum(dim=1).to(torch.int32).repeat_interleave(8)
    )

    expected[9, :_BLOCK_THREADS] = warp_exclusive.reshape(-1)
    expected[10, :_BLOCK_THREADS] = warp_inclusive.reshape(-1)
    expected[11, :_BLOCK_THREADS] = warp_exclusive.reshape(-1)
    expected[12, :_BLOCK_THREADS] = warp_inclusive.reshape(-1)
    expected[13, :_BLOCK_THREADS] = warp_exclusive_max.reshape(-1)
    expected[14, :_BLOCK_THREADS] = warp_inclusive_max.reshape(-1)
    expected[15, :_BLOCK_THREADS] = warp_totals
    expected[16, :_BLOCK_THREADS] = subgroup_totals
    expected[17, :_BLOCK_THREADS] = values[:_BLOCK_THREADS]
    expected[18, 0] = values[:_BLOCK_THREADS].max()
    expected[19, 0] = warp_values[0, :24].max()
    expected[19, 32] = warp_values[1, :24].max()
    return expected.reshape(-1)


@pytest.mark.evidence_for("group.reduce", backend="cutlass", evidence="runtime")
@pytest.mark.evidence_for("group.sum", backend="cutlass", evidence="runtime")
@pytest.mark.evidence_for("group.scan", backend="cutlass", evidence="runtime")
@pytest.mark.evidence_for("group.exclusive_sum", backend="cutlass", evidence="runtime")
@pytest.mark.evidence_for("group.inclusive_sum", backend="cutlass", evidence="runtime")
@pytest.mark.evidence_for("group.exclusive_scan", backend="cutlass", evidence="runtime")
@pytest.mark.evidence_for("group.inclusive_scan", backend="cutlass", evidence="runtime")
def test_common_reduce_sum_scan_matches_qualified_cutlass_and_oracle() -> None:
    cutlass.cuda.initialize_cuda_context()
    values_host = (torch.arange(_ITEM_COUNT, dtype=torch.int32) * 17 % 97) - 48
    values = values_host.cuda()
    common_output = torch.full(
        (_SEGMENT_COUNT * _ITEM_COUNT,),
        _SENTINEL,
        dtype=torch.int32,
        device="cuda",
    )
    qualified_output = torch.full_like(common_output, _SENTINEL)

    _run_common_sum_scan(from_dlpack(values), from_dlpack(common_output))
    _run_qualified_sum_scan(from_dlpack(values), from_dlpack(qualified_output))
    torch.cuda.synchronize()

    torch.testing.assert_close(
        common_output,
        qualified_output,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        common_output.cpu(),
        _expected_output(values_host),
        atol=0,
        rtol=0,
    )
