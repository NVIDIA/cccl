# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.e2e import (
    configure_e2e_environment as _configure_e2e_environment,
)
from ..support.e2e import import_e2e_modules as _load_e2e_modules


def _import_e2e_modules():
    global ct, coop
    torch, ct, coop = _load_e2e_modules()
    return torch, ct, coop


def _expected_head_flags(torch, input_2d):
    flags = torch.zeros_like(input_2d, dtype=torch.int32)
    flags[:, 0] = 1
    flags[:, 1:] = (input_2d[:, 1:] != input_2d[:, :-1]).to(torch.int32)
    return flags


def _expected_tail_flags(torch, input_2d):
    flags = torch.zeros_like(input_2d, dtype=torch.int32)
    flags[:, :-1] = (input_2d[:, :-1] != input_2d[:, 1:]).to(torch.int32)
    flags[:, -1] = 1
    return flags


def test_cuda_coop_tile_executes_discontinuity_flags(monkeypatch):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 128
    indices = torch.arange(num_items, device="cuda", dtype=torch.int32)
    first = indices // 4
    second = (indices + 2) // 5
    second[0] = -17
    second[-1] = 31337
    input_2d = torch.stack((first, second))
    input_data = input_2d.flatten()
    paired_heads_out = torch.zeros_like(input_data)
    paired_tails_out = torch.zeros_like(input_data)
    heads_only_out = torch.zeros_like(input_data)
    tails_only_out = torch.zeros_like(input_data)

    @ct.kernel
    def kernel(
        inp: ct.Array,
        paired_heads_arr: ct.Array,
        paired_tails_arr: ct.Array,
        heads_only_arr: ct.Array,
        tails_only_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        values = ct.load(inp, (batch_idx,), shape=(num_items,))
        paired_heads, paired_tails = coop.discontinuity_flag_heads_and_tails(
            values
        )
        heads_only = coop.discontinuity_flag_heads(values)
        tails_only = coop.discontinuity_flag_tails(values)
        ct.store(paired_heads_arr, (batch_idx,), paired_heads)
        ct.store(paired_tails_arr, (batch_idx,), paired_tails)
        ct.store(heads_only_arr, (batch_idx,), heads_only)
        ct.store(tails_only_arr, (batch_idx,), tails_only)

    ct.launch(
        torch.cuda.current_stream(),
        (input_2d.shape[0], 1, 1),
        kernel,
        (
            input_data,
            paired_heads_out,
            paired_tails_out,
            heads_only_out,
            tails_only_out,
        ),
    )
    torch.cuda.synchronize()

    expected_heads = _expected_head_flags(torch, input_2d)
    expected_tails = _expected_tail_flags(torch, input_2d)
    assert torch.equal(paired_heads_out.view_as(expected_heads), expected_heads)
    assert torch.equal(heads_only_out.view_as(expected_heads), expected_heads)
    assert torch.equal(paired_tails_out.view_as(expected_tails), expected_tails)
    assert torch.equal(tails_only_out.view_as(expected_tails), expected_tails)


@pytest.mark.parametrize("dtype_name", ["uint8", "int64", "float32", "float16"])
def test_cuda_coop_tile_executes_discontinuity_and_adjacent_difference(
    monkeypatch,
    dtype_name,
):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 256
    batch_size = 2
    groups = torch.arange(num_items, device="cuda", dtype=torch.int32)
    groups = (groups // 4).repeat(batch_size, 1)
    diffs = torch.arange(num_items, device="cuda", dtype=torch.int32)
    diffs = (diffs * 3 + 1).repeat(batch_size, 1)
    dtype = getattr(torch, dtype_name)
    groups = groups.to(dtype)
    diffs = diffs.to(dtype)
    if dtype_name.startswith("float"):
        groups[:, 30] = float("nan")
    group_data = groups.flatten()
    diff_data = diffs.flatten()
    heads_out = torch.zeros_like(group_data, dtype=torch.int32)
    tails_out = torch.zeros_like(group_data, dtype=torch.int32)
    heads_only_out = torch.zeros_like(group_data, dtype=torch.int32)
    tails_only_out = torch.zeros_like(group_data, dtype=torch.int32)
    left_out = torch.zeros_like(diff_data)
    right_out = torch.zeros_like(diff_data)

    up_out = torch.empty_like(diff_data)
    down_out = torch.empty_like(diff_data)
    original_groups = torch.empty_like(group_data)
    original_diffs = torch.empty_like(diff_data)

    @ct.kernel(num_worker_warps=4)
    def kernel(
        groups_in: ct.Array,
        diffs_in: ct.Array,
        heads_arr: ct.Array,
        tails_arr: ct.Array,
        heads_only_arr: ct.Array,
        tails_only_arr: ct.Array,
        left_arr: ct.Array,
        right_arr: ct.Array,
        up_arr: ct.Array,
        down_arr: ct.Array,
        preserved_groups: ct.Array,
        preserved_diffs: ct.Array,
    ):
        batch_idx = ct.bid(0)
        group_tile = ct.load(groups_in, (batch_idx,), shape=(num_items,))
        diff_tile = ct.load(diffs_in, (batch_idx,), shape=(num_items,))
        heads, tails = coop.discontinuity_flag_heads_and_tails(group_tile)
        heads_only = coop.discontinuity_flag_heads(group_tile)
        tails_only = coop.discontinuity_flag_tails(group_tile)
        left = coop.adjacent_difference_subtract_left(diff_tile)
        right = coop.adjacent_difference_subtract_right(diff_tile)
        up = coop.shuffle_up(diff_tile)
        down = coop.shuffle_down(diff_tile)
        ct.store(heads_arr, (batch_idx,), heads)
        ct.store(tails_arr, (batch_idx,), tails)
        ct.store(heads_only_arr, (batch_idx,), heads_only)
        ct.store(tails_only_arr, (batch_idx,), tails_only)
        ct.store(left_arr, (batch_idx,), left)
        ct.store(right_arr, (batch_idx,), right)
        ct.store(up_arr, (batch_idx,), up)
        ct.store(down_arr, (batch_idx,), down)
        ct.store(preserved_groups, (batch_idx,), group_tile)
        ct.store(preserved_diffs, (batch_idx,), diff_tile)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (
            group_data,
            diff_data,
            heads_out,
            tails_out,
            heads_only_out,
            tails_only_out,
            left_out,
            right_out,
            up_out,
            down_out,
            original_groups,
            original_diffs,
        ),
    )
    torch.cuda.synchronize()

    expected_left = torch.empty_like(diffs)
    expected_left[:, 0] = diffs[:, 0]
    expected_left[:, 1:] = diffs[:, 1:] - diffs[:, :-1]
    expected_right = torch.empty_like(diffs)
    expected_right[:, :-1] = diffs[:, :-1] - diffs[:, 1:]
    expected_right[:, -1] = diffs[:, -1]

    assert torch.equal(
        heads_out.view_as(groups), _expected_head_flags(torch, groups)
    )
    assert torch.equal(
        tails_out.view_as(groups), _expected_tail_flags(torch, groups)
    )
    assert torch.equal(
        heads_only_out.view_as(groups),
        _expected_head_flags(torch, groups),
    )
    assert torch.equal(
        tails_only_out.view_as(groups),
        _expected_tail_flags(torch, groups),
    )
    assert torch.equal(left_out.view_as(expected_left), expected_left)
    assert torch.equal(right_out.view_as(expected_right), expected_right)

    expected_up = torch.cat((diffs[:, :1], diffs[:, :-1]), dim=1)
    expected_down = torch.cat((diffs[:, 1:], diffs[:, -1:]), dim=1)
    assert torch.equal(up_out.view_as(diffs), expected_up)
    assert torch.equal(down_out.view_as(diffs), expected_down)
    torch.testing.assert_close(
        original_groups.view_as(groups), groups, rtol=0, atol=0, equal_nan=True
    )
    assert torch.equal(original_diffs.view_as(diffs), diffs)
