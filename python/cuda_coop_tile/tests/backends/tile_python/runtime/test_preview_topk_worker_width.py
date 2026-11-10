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


def _sorted_pairs(keys, values):
    return sorted(zip(keys.cpu().tolist(), values.cpu().tolist(), strict=True))


@pytest.mark.parametrize(
    ("num_items", "simt_num_warps"),
    [(64, 2)],
)
def test_cuda_coop_tile_executes_topk_score_windows(
    monkeypatch,
    num_items,
    simt_num_warps,
):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    batch_size = 2
    k = 9
    valid_items = num_items - 11
    base = torch.arange(num_items, device="cuda", dtype=torch.int32)
    keys_2d = ((base * 37 + 11) % 251).repeat(batch_size, 1)
    keys_2d[1] = (keys_2d[1] + 5) % 251
    values_2d = (base * 13 + 5).repeat(batch_size, 1)
    values_2d = values_2d + (
        torch.arange(batch_size, device="cuda", dtype=torch.int32).unsqueeze(1)
        * 10000
    )

    keys_in = keys_2d.flatten()
    values_in = values_2d.flatten()
    k_in = torch.tensor([k], device="cuda", dtype=torch.int32)
    max_keys_out = torch.zeros_like(keys_in)
    min_pair_keys_out = torch.zeros_like(keys_in)
    min_pair_values_out = torch.zeros_like(values_in)

    @ct.kernel(num_worker_warps=simt_num_warps)
    def kernel(
        keys_arr: ct.Array,
        values_arr: ct.Array,
        k_arr: ct.Array,
        max_keys_arr: ct.Array,
        min_pair_keys_arr: ct.Array,
        min_pair_values_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        keys = ct.load(keys_arr, (batch_idx,), shape=(num_items,))
        values = ct.load(values_arr, (batch_idx,), shape=(num_items,))
        runtime_k = ct.load(k_arr, (0,), shape=())
        top_keys = coop.topk_max_keys(
            keys,
            k=runtime_k,
            begin_bit=0,
            end_bit=8,
        )
        pair_keys, pair_values = coop.topk_min_pairs(
            keys,
            values,
            k=runtime_k,
            valid_items=valid_items,
            begin_bit=0,
            end_bit=8,
        )
        ct.store(max_keys_arr, (batch_idx,), top_keys)
        ct.store(min_pair_keys_arr, (batch_idx,), pair_keys)
        ct.store(min_pair_values_arr, (batch_idx,), pair_values)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (
            keys_in,
            values_in,
            k_in,
            max_keys_out,
            min_pair_keys_out,
            min_pair_values_out,
        ),
    )
    torch.cuda.synchronize()

    expected_max_keys = torch.sort(keys_2d, dim=1, descending=True).values
    expected_min_keys, min_idx = torch.sort(keys_2d[:, :valid_items], dim=1)
    expected_min_values = torch.gather(values_2d[:, :valid_items], 1, min_idx)
    max_keys_out_2d = max_keys_out.view(batch_size, num_items)
    min_pair_keys_out_2d = min_pair_keys_out.view(batch_size, num_items)
    min_pair_values_out_2d = min_pair_values_out.view(batch_size, num_items)

    for batch_idx in range(batch_size):
        assert torch.equal(
            torch.sort(max_keys_out_2d[batch_idx, :k]).values,
            torch.sort(expected_max_keys[batch_idx, :k]).values,
        )
        assert _sorted_pairs(
            min_pair_keys_out_2d[batch_idx, :k],
            min_pair_values_out_2d[batch_idx, :k],
        ) == _sorted_pairs(
            expected_min_keys[batch_idx, :k],
            expected_min_values[batch_idx, :k],
        )
