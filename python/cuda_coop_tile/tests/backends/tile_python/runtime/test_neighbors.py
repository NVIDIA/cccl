# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from ..support.e2e import (
    configure_e2e_environment as _configure_e2e_environment,
)
from ..support.e2e import import_e2e_modules as _load_e2e_modules


def _import_e2e_modules():
    global ct, coop
    torch, ct, coop = _load_e2e_modules()
    return torch, ct, coop


def test_cuda_coop_tile_executes_adjacent_difference_neighbors(monkeypatch):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 128
    batch_size = 2
    diffs = torch.arange(num_items, device="cuda", dtype=torch.int32)
    diffs = (diffs * 3 + 1).repeat(batch_size, 1)
    diff_data = diffs.flatten()
    left_out = torch.zeros_like(diff_data)
    right_out = torch.zeros_like(diff_data)

    @ct.kernel
    def kernel(diffs_in: ct.Array, left_arr: ct.Array, right_arr: ct.Array):
        batch_idx = ct.bid(0)
        diff_tile = ct.load(diffs_in, (batch_idx,), shape=(num_items,))
        left = coop.adjacent_difference_subtract_left(diff_tile)
        right = coop.adjacent_difference_subtract_right(diff_tile)
        ct.store(left_arr, (batch_idx,), left)
        ct.store(right_arr, (batch_idx,), right)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (diff_data, left_out, right_out),
    )
    torch.cuda.synchronize()

    expected_left = torch.empty_like(diffs)
    expected_left[:, 0] = diffs[:, 0]
    expected_left[:, 1:] = diffs[:, 1:] - diffs[:, :-1]
    expected_right = torch.empty_like(diffs)
    expected_right[:, :-1] = diffs[:, :-1] - diffs[:, 1:]
    expected_right[:, -1] = diffs[:, -1]
    assert torch.equal(left_out.view_as(expected_left), expected_left)
    assert torch.equal(right_out.view_as(expected_right), expected_right)


@pytest.mark.parametrize("num_items", [128, 512, 1024])
def test_cuda_coop_tile_executes_shuffle_neighbors(monkeypatch, num_items):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    batch_size = 2
    rotate_distance = 3
    offset_distance = 2
    shuffle_2d = torch.arange(num_items, device="cuda", dtype=torch.int32)
    shuffle_2d = shuffle_2d.repeat(batch_size, 1)
    shuffle_data = shuffle_2d.flatten()
    up_out = torch.zeros_like(shuffle_data)
    down_out = torch.zeros_like(shuffle_data)
    offset_out = torch.zeros_like(shuffle_data)
    rotate_out = torch.zeros_like(shuffle_data)

    @ct.kernel(num_worker_warps=num_items // 32)
    def kernel(
        shuffle_arr: ct.Array,
        up_arr: ct.Array,
        down_arr: ct.Array,
        offset_arr: ct.Array,
        rotate_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        shuffle_tile = ct.load(shuffle_arr, (batch_idx,), shape=(num_items,))
        up = coop.shuffle_up(shuffle_tile)
        down = coop.shuffle_down(shuffle_tile)
        offset = coop.shuffle_offset(shuffle_tile, distance=offset_distance)
        rotated = coop.shuffle_rotate(shuffle_tile, distance=rotate_distance)
        ct.store(up_arr, (batch_idx,), up)
        ct.store(down_arr, (batch_idx,), down)
        ct.store(offset_arr, (batch_idx,), offset)
        ct.store(rotate_arr, (batch_idx,), rotated)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (shuffle_data, up_out, down_out, offset_out, rotate_out),
    )
    torch.cuda.synchronize()

    expected_up = shuffle_2d.clone()
    expected_up[:, 1:] = shuffle_2d[:, :-1]
    expected_down = shuffle_2d.clone()
    expected_down[:, :-1] = shuffle_2d[:, 1:]
    expected_rotate = torch.roll(shuffle_2d, shifts=-rotate_distance, dims=1)
    assert torch.equal(up_out.view_as(expected_up), expected_up)
    assert torch.equal(down_out.view_as(expected_down), expected_down)
    expected_offset = shuffle_2d.clone()
    expected_offset[:, :-offset_distance] = shuffle_2d[:, offset_distance:]
    assert torch.equal(offset_out.view_as(expected_offset), expected_offset)
    assert torch.equal(rotate_out.view_as(expected_rotate), expected_rotate)
