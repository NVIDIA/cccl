# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib

import pytest

from ....support.paths import PACKAGE_ROOT
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


def _expected_integer_radix_order(
    torch,
    keys,
    values,
    *,
    begin_bit,
    end_bit,
    descending,
):
    width_bits = 64 if keys.dtype in {torch.int64, torch.uint64} else 32
    full_mask = (1 << width_bits) - 1
    raw_unsigned = [(int(key) & full_mask) for key in keys.cpu().tolist()]
    sign_flip = (
        0
        if keys.dtype in {torch.uint32, torch.uint64}
        else 1 << (width_bits - 1)
    )
    ordered = [value ^ sign_flip for value in raw_unsigned]
    mask = (1 << (end_bit - begin_bit)) - 1
    digits = [int((value >> begin_bit) & mask) for value in ordered]
    indices = list(range(len(digits)))
    if descending:
        indices.sort(key=lambda index: (-digits[index], index))
    else:
        indices.sort(key=lambda index: (digits[index], index))
    index = torch.tensor(indices, dtype=torch.long)
    return keys.cpu()[index], values.cpu()[index]


@pytest.mark.parametrize(
    ("num_items", "simt_num_warps"),
    [(128, 4)],
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


@pytest.mark.parametrize("digit_bits,k", [(12, 11), (3, 45)])
def test_cuda_coop_tile_executes_multi_item_runtime_topk_controls(
    monkeypatch, digit_bits, k
):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 256
    valid_items = 243
    begin_bit = 33
    end_bit = begin_bit + digit_bits
    base = torch.arange(num_items, device="cuda", dtype=torch.int64)
    radix_digits = (base * 37) % (1 << (end_bit - begin_bit))
    keys_in = (radix_digits << begin_bit) | (
        (base * 13 + 5) & ((1 << begin_bit) - 1)
    )
    values_in = base.to(torch.float64) * 1.25 + 0.5
    controls = torch.tensor(
        [k, valid_items, begin_bit, end_bit],
        device="cuda",
        dtype=torch.int32,
    )
    min_keys_out = torch.zeros_like(keys_in)
    max_pair_keys_out = torch.zeros_like(keys_in)
    max_pair_values_out = torch.zeros_like(values_in)

    @ct.kernel(num_worker_warps=4)
    def kernel(
        keys_arr: ct.Array,
        values_arr: ct.Array,
        controls_arr: ct.Array,
        min_keys_arr: ct.Array,
        max_pair_keys_arr: ct.Array,
        max_pair_values_arr: ct.Array,
    ):
        keys = ct.load(keys_arr, (0,), shape=(num_items,))
        values = ct.load(values_arr, (0,), shape=(num_items,))
        runtime_k = ct.load(controls_arr, (0,), shape=())
        runtime_num_valid = ct.load(controls_arr, (1,), shape=())
        runtime_begin_bit = ct.load(controls_arr, (2,), shape=())
        runtime_end_bit = ct.load(controls_arr, (3,), shape=())
        min_keys = coop.topk_min_keys(
            keys,
            k=runtime_k,
            valid_items=runtime_num_valid,
            begin_bit=runtime_begin_bit,
            end_bit=runtime_end_bit,
        )
        max_pair_keys, max_pair_values = coop.topk_max_pairs(
            keys,
            values,
            k=runtime_k,
            valid_items=runtime_num_valid,
            begin_bit=runtime_begin_bit,
            end_bit=runtime_end_bit,
        )
        ct.store(min_keys_arr, (0,), min_keys)
        ct.store(max_pair_keys_arr, (0,), max_pair_keys)
        ct.store(max_pair_values_arr, (0,), max_pair_values)

    ct.launch(
        torch.cuda.current_stream(),
        (1, 1, 1),
        kernel,
        (
            keys_in,
            values_in,
            controls,
            min_keys_out,
            max_pair_keys_out,
            max_pair_values_out,
        ),
    )
    torch.cuda.synchronize()

    expected_min_keys, _ = _expected_integer_radix_order(
        torch,
        keys_in[:valid_items],
        values_in[:valid_items],
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=False,
    )
    expected_max_keys, _ = _expected_integer_radix_order(
        torch,
        keys_in[:valid_items],
        values_in[:valid_items],
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=True,
    )
    # The three-bit case cuts through a tie bucket. Compare selected digits
    # and source association without requiring a particular tied item/order.
    mask = (1 << digit_bits) - 1
    for selected, expected in (
        (min_keys_out[:k], expected_min_keys[:k]),
        (max_pair_keys_out[:k], expected_max_keys[:k]),
    ):
        actual_keys = selected.cpu().tolist()
        assert len(set(actual_keys)) == k
        assert set(actual_keys) <= set(keys_in[:valid_items].cpu().tolist())
        assert sorted(
            (key >> begin_bit) & mask for key in actual_keys
        ) == sorted((key >> begin_bit) & mask for key in expected.tolist())
    source_pairs = dict(
        zip(
            keys_in[:valid_items].cpu().tolist(),
            values_in[:valid_items].cpu().tolist(),
            strict=True,
        )
    )
    assert max_pair_values_out[:k].cpu().tolist() == [
        source_pairs[key] for key in max_pair_keys_out[:k].cpu().tolist()
    ]


@pytest.mark.parametrize("num_items", [128, 512])
def test_cuda_coop_tile_topk_empty_and_clamped_windows(monkeypatch, num_items):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    keys = (
        (torch.arange(num_items, device="cuda", dtype=torch.int32) * 73)
        % num_items
    ) - num_items // 2
    values = torch.arange(num_items, device="cuda", dtype=torch.float64) * 1.25
    original_keys = keys.clone()
    original_values = values.clone()
    controls = torch.empty(2, device="cuda", dtype=torch.int32)
    min_keys_out = torch.empty_like(keys)
    max_keys_out = torch.empty_like(keys)
    min_pair_keys = torch.empty_like(keys)
    max_pair_keys = torch.empty_like(keys)
    min_pair_values = torch.empty_like(values)
    max_pair_values = torch.empty_like(values)

    @ct.kernel(num_worker_warps=4)
    def kernel(
        keys_arr: ct.Array,
        values_arr: ct.Array,
        controls_arr: ct.Array,
        min_keys_arr: ct.Array,
        max_keys_arr: ct.Array,
        min_pair_keys_arr: ct.Array,
        min_pair_values_arr: ct.Array,
        max_pair_keys_arr: ct.Array,
        max_pair_values_arr: ct.Array,
    ):
        key_tile = ct.load(keys_arr, (0,), shape=(num_items,))
        value_tile = ct.load(values_arr, (0,), shape=(num_items,))
        k = ct.load(controls_arr, (0,), shape=())
        valid_items = ct.load(controls_arr, (1,), shape=())
        minimum = coop.topk_min_keys(key_tile, k=k, valid_items=valid_items)
        maximum = coop.topk_max_keys(key_tile, k=k, valid_items=valid_items)
        min_keys, min_values = coop.topk_min_pairs(
            key_tile, value_tile, k=k, valid_items=valid_items
        )
        max_keys, max_values = coop.topk_max_pairs(
            key_tile, value_tile, k=k, valid_items=valid_items
        )
        ct.store(min_keys_arr, (0,), minimum)
        ct.store(max_keys_arr, (0,), maximum)
        ct.store(min_pair_keys_arr, (0,), min_keys)
        ct.store(min_pair_values_arr, (0,), min_values)
        ct.store(max_pair_keys_arr, (0,), max_keys)
        ct.store(max_pair_values_arr, (0,), max_values)
        ct.store(keys_arr, (0,), key_tile)
        ct.store(values_arr, (0,), value_tile)

    for k, valid_items in [
        (0, num_items),
        (5, 0),
        (0, 0),
        (num_items, 11),
        (17, num_items),
        (num_items, num_items),
    ]:
        controls.copy_(
            torch.tensor([k, valid_items], device="cuda", dtype=torch.int32)
        )
        ct.launch(
            torch.cuda.current_stream(),
            (1, 1, 1),
            kernel,
            (
                keys,
                values,
                controls,
                min_keys_out,
                max_keys_out,
                min_pair_keys,
                min_pair_values,
                max_pair_keys,
                max_pair_values,
            ),
        )
        torch.cuda.synchronize()
        selected = min(k, valid_items)
        for descending, key_output, pair_keys, pair_values in [
            (False, min_keys_out, min_pair_keys, min_pair_values),
            (True, max_keys_out, max_pair_keys, max_pair_values),
        ]:
            order = torch.argsort(
                original_keys[:valid_items], descending=descending
            )[:selected]
            expected_keys = original_keys[order]
            expected_values = original_values[order]
            assert torch.equal(
                torch.sort(key_output[:selected]).values,
                torch.sort(expected_keys).values,
            )
            assert _sorted_pairs(
                pair_keys[:selected], pair_values[:selected]
            ) == _sorted_pairs(expected_keys, expected_values)
        assert torch.equal(keys, original_keys)
        assert torch.equal(values, original_values)


def test_cuda_coop_tile_executes_mma_row_topk_pipeline(monkeypatch):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    mma_topk = importlib.import_module("examples.mma_topk")

    rows = mma_topk.M_TILE
    inner = mma_topk.K_TILE
    columns = mma_topk.N_TILE * mma_topk.COLUMN_TILES
    bit_rows = torch.arange(inner, device="cuda", dtype=torch.int64).unsqueeze(
        1
    )
    column_ids = torch.arange(columns, device="cuda", dtype=torch.int64)
    b_array = ((column_ids.unsqueeze(0) >> bit_rows) & 1).to(torch.float32)
    weights = torch.zeros(inner, device="cuda", dtype=torch.float32)
    weights[:9] = 2.0 ** torch.arange(9, device="cuda", dtype=torch.float32)
    a_array = weights.unsqueeze(0).repeat(rows, 1)

    partial_scores = torch.empty(
        (rows, mma_topk.CANDIDATES_PER_ROW), device="cuda", dtype=torch.float32
    )
    partial_columns = torch.empty(
        (rows, mma_topk.CANDIDATES_PER_ROW), device="cuda", dtype=torch.int32
    )
    top_scores = torch.empty(
        (rows, mma_topk.TOPK), device="cuda", dtype=torch.float32
    )
    top_columns = torch.empty(
        (rows, mma_topk.TOPK), device="cuda", dtype=torch.int32
    )

    mma_topk.launch_mma_topk(
        torch.cuda.current_stream(),
        a_array,
        b_array,
        partial_scores,
        partial_columns,
        top_scores,
        top_columns,
    )
    torch.cuda.synchronize()

    expected_scores, expected_columns = torch.topk(
        a_array @ b_array, mma_topk.TOPK, dim=1
    )
    actual_order = torch.argsort(top_columns, dim=1)
    expected_order = torch.argsort(expected_columns, dim=1)
    assert torch.equal(
        torch.gather(top_columns, 1, actual_order),
        torch.gather(expected_columns.to(torch.int32), 1, expected_order),
    )
    assert torch.equal(
        torch.gather(top_scores, 1, actual_order),
        torch.gather(expected_scores, 1, expected_order),
    )
