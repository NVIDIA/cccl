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


def test_cuda_coop_tile_executes_warp_sort_in_four_warp_cta(monkeypatch):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 32
    batch_size = 2
    keys_2d = torch.tensor(
        [
            17,
            3,
            29,
            3,
            11,
            31,
            0,
            23,
            7,
            19,
            5,
            13,
            2,
            28,
            9,
            21,
            16,
            1,
            30,
            4,
            10,
            27,
            6,
            22,
            8,
            20,
            12,
            26,
            14,
            24,
            15,
            25,
        ],
        device="cuda",
        dtype=torch.int32,
    ).repeat(batch_size, 1)
    keys_2d[1] = torch.flip(keys_2d[1], dims=(0,))
    keys_in = keys_2d.flatten()
    keys_out = torch.zeros_like(keys_in)

    @ct.kernel(num_worker_warps=4)
    def kernel(keys_arr: ct.Array, output_arr: ct.Array):
        batch_idx = ct.bid(0)
        keys = ct.load(keys_arr, (batch_idx,), shape=(num_items,))
        sorted_keys = coop.merge_sort_keys(keys)
        ct.store(output_arr, (batch_idx,), sorted_keys)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (keys_in, keys_out),
    )
    torch.cuda.synchronize()

    expected = torch.sort(keys_2d, dim=1).values
    assert torch.equal(keys_out.view_as(expected), expected)


def test_cuda_coop_tile_executes_merge_sort_keys_full_and_partial(monkeypatch):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 128
    valid_items = 96
    batch_size = 2
    oob_default = -10_000.0
    full_keys_2d = torch.arange(
        num_items,
        0,
        -1,
        device="cuda",
        dtype=torch.float32,
    ).repeat(batch_size, 1)
    valid_keys = torch.arange(
        1,
        valid_items + 1,
        device="cuda",
        dtype=torch.float32,
    )
    invalid_keys = torch.full(
        (num_items - valid_items,),
        -123.0,
        device="cuda",
        dtype=torch.float32,
    )
    partial_keys_2d = torch.cat((valid_keys, invalid_keys)).repeat(
        batch_size, 1
    )

    full_keys_in = full_keys_2d.flatten()
    partial_keys_in = partial_keys_2d.flatten()
    full_keys_out = torch.zeros_like(full_keys_in)
    partial_keys_out = torch.zeros_like(partial_keys_in)

    @ct.kernel
    def kernel(
        full_keys_arr: ct.Array,
        partial_keys_arr: ct.Array,
        full_keys_out_arr: ct.Array,
        partial_keys_out_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        full_keys = ct.load(full_keys_arr, (batch_idx,), shape=(num_items,))
        partial_keys = ct.load(
            partial_keys_arr, (batch_idx,), shape=(num_items,)
        )
        sorted_full_keys = coop.merge_sort_keys(full_keys)
        sorted_partial_keys = coop.merge_sort_keys(
            partial_keys,
            descending=True,
            valid_items=valid_items,
            oob_default=oob_default,
        )
        ct.store(full_keys_out_arr, (batch_idx,), sorted_full_keys)
        ct.store(partial_keys_out_arr, (batch_idx,), sorted_partial_keys)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (
            full_keys_in,
            partial_keys_in,
            full_keys_out,
            partial_keys_out,
        ),
    )
    torch.cuda.synchronize()

    expected_full = torch.sort(full_keys_2d, dim=1).values
    assert torch.allclose(
        full_keys_out.view_as(expected_full),
        expected_full,
    )
    partial_out_2d = partial_keys_out.view(batch_size, num_items)
    expected_valid = torch.arange(
        valid_items,
        0,
        -1,
        device="cuda",
        dtype=torch.float32,
    ).repeat(batch_size, 1)
    assert torch.allclose(
        partial_out_2d[:, :valid_items],
        expected_valid,
    )
    assert torch.allclose(
        partial_out_2d[:, valid_items:],
        torch.full_like(partial_out_2d[:, valid_items:], oob_default),
    )


@pytest.mark.parametrize(
    "key_dtype,value_dtype",
    [("float32", "int32"), ("float16", "float16"), ("bfloat16", "bfloat16")],
)
def test_cuda_coop_tile_executes_merge_sort_pairs_full_and_partial(
    monkeypatch, key_dtype, value_dtype
):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 256
    valid_items = 193
    key_dtype = getattr(torch, key_dtype)
    value_dtype = getattr(torch, value_dtype)
    batch_size = 2
    oob_default = -10_000.0
    full_keys_2d = torch.arange(
        num_items,
        0,
        -1,
        device="cuda",
        dtype=key_dtype,
    ).repeat(batch_size, 1)
    full_vals_2d = torch.arange(
        num_items,
        device="cuda",
        dtype=value_dtype,
    ).repeat(batch_size, 1)
    valid_keys = torch.arange(
        1,
        valid_items + 1,
        device="cuda",
        dtype=key_dtype,
    )
    valid_keys = valid_keys % 17
    invalid_keys = torch.full(
        (num_items - valid_items,),
        -123.0,
        device="cuda",
        dtype=key_dtype,
    )
    partial_keys_2d = torch.cat((valid_keys, invalid_keys)).repeat(
        batch_size, 1
    )
    partial_vals_2d = torch.arange(
        num_items,
        device="cuda",
        dtype=value_dtype,
    ).repeat(batch_size, 1)

    full_keys_in = full_keys_2d.flatten()
    full_vals_in = full_vals_2d.flatten()
    partial_keys_in = partial_keys_2d.flatten()
    partial_vals_in = partial_vals_2d.flatten()
    full_keys_out = torch.zeros_like(full_keys_in)
    full_vals_out = torch.zeros_like(full_vals_in)
    partial_keys_out = torch.zeros_like(partial_keys_in)
    partial_vals_out = torch.zeros_like(partial_vals_in)

    @ct.kernel(num_worker_warps=4)
    def kernel(
        full_keys_arr: ct.Array,
        full_vals_arr: ct.Array,
        partial_keys_arr: ct.Array,
        partial_vals_arr: ct.Array,
        full_keys_out_arr: ct.Array,
        full_vals_out_arr: ct.Array,
        partial_keys_out_arr: ct.Array,
        partial_vals_out_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        full_keys = ct.load(full_keys_arr, (batch_idx,), shape=(num_items,))
        full_vals = ct.load(full_vals_arr, (batch_idx,), shape=(num_items,))
        partial_keys = ct.load(
            partial_keys_arr, (batch_idx,), shape=(num_items,)
        )
        partial_vals = ct.load(
            partial_vals_arr, (batch_idx,), shape=(num_items,)
        )
        sorted_full_keys, sorted_full_vals = coop.merge_sort_pairs(
            full_keys,
            full_vals,
        )
        sorted_partial_keys, sorted_partial_vals = coop.merge_sort_pairs(
            partial_keys,
            partial_vals,
            descending=True,
            valid_items=valid_items,
            oob_default=oob_default,
        )
        ct.store(full_keys_out_arr, (batch_idx,), sorted_full_keys)
        ct.store(full_vals_out_arr, (batch_idx,), sorted_full_vals)
        ct.store(partial_keys_out_arr, (batch_idx,), sorted_partial_keys)
        ct.store(partial_vals_out_arr, (batch_idx,), sorted_partial_vals)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (
            full_keys_in,
            full_vals_in,
            partial_keys_in,
            partial_vals_in,
            full_keys_out,
            full_vals_out,
            partial_keys_out,
            partial_vals_out,
        ),
    )
    torch.cuda.synchronize()

    expected_full_keys, full_indices = torch.sort(full_keys_2d, dim=1)
    expected_full_vals = torch.gather(full_vals_2d, 1, full_indices)
    assert torch.allclose(
        full_keys_out.view_as(expected_full_keys),
        expected_full_keys,
    )
    assert torch.equal(
        full_vals_out.view_as(expected_full_vals),
        expected_full_vals,
    )

    partial_keys_out_2d = partial_keys_out.view(batch_size, num_items)
    partial_vals_out_2d = partial_vals_out.view(batch_size, num_items)
    expected_valid = torch.sort(valid_keys, descending=True).values
    assert torch.equal(
        partial_keys_out_2d[:, :valid_items],
        expected_valid.unsqueeze(0).repeat(batch_size, 1),
    )
    # Equal keys have no stability guarantee; compare key/payload multisets.
    expected_pairs = sorted(zip(valid_keys.cpu().tolist(), range(valid_items)))
    for keys, values in zip(
        partial_keys_out_2d[:, :valid_items].cpu().tolist(),
        partial_vals_out_2d[:, :valid_items].cpu().tolist(),
    ):
        assert sorted(zip(keys, values)) == expected_pairs
    assert torch.allclose(
        partial_keys_out_2d[:, valid_items:],
        torch.full_like(partial_keys_out_2d[:, valid_items:], oob_default),
    )
    assert torch.equal(
        partial_vals_out_2d[:, valid_items:],
        torch.zeros_like(partial_vals_out_2d[:, valid_items:]),
    )
