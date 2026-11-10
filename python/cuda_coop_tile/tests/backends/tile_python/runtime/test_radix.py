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


def _validate_radix_ranks(
    torch, keys_2d, ranks_2d, begin_bit, end_bit, descending
):
    radix_bits = end_bit - begin_bit
    radix_digits = 1 << radix_bits
    mask = radix_digits - 1
    expected_ranks = torch.arange(
        keys_2d.shape[1],
        device=keys_2d.device,
        dtype=ranks_2d.dtype,
    )

    for keys, ranks in zip(keys_2d, ranks_2d, strict=True):
        digits = (keys >> begin_bit) & mask
        counts = torch.bincount(
            digits.to(torch.int64),
            minlength=radix_digits,
        )
        offsets = {}
        running = 0
        digit_range = (
            range(radix_digits - 1, -1, -1)
            if descending
            else range(radix_digits)
        )
        for digit in digit_range:
            offsets[digit] = running
            running += int(counts[digit])

        assert torch.equal(torch.sort(ranks).values, expected_ranks)
        for digit, rank in zip(digits.tolist(), ranks.tolist(), strict=True):
            start = offsets[digit]
            end = start + int(counts[digit])
            assert start <= rank < end


@pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32", "int64"])
@pytest.mark.parametrize("num_items", [128, 512])
def test_radix_rank_keys_signed_order_and_stable_ties(
    monkeypatch, dtype_name, num_items
):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()
    dtype = getattr(torch, dtype_name)
    bits = torch.iinfo(dtype).bits
    minimum, maximum = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    values = [minimum, 0, maximum, -1, 1, minimum, -1, 0]
    keys = torch.tensor(
        values * (num_items // len(values)), device="cuda", dtype=dtype
    )
    preserved = keys.clone()
    ascending = torch.empty(num_items, device="cuda", dtype=torch.int32)
    descending = torch.empty_like(ascending)
    begin_bit = bits - 1

    @ct.kernel(num_worker_warps=4)
    def kernel(source: ct.Array, asc: ct.Array, desc: ct.Array):
        tile = ct.load(source, (0,), shape=(num_items,))
        asc_ranks = coop.radix_rank_keys(tile, begin_bit=begin_bit)
        desc_ranks = coop.radix_rank_keys(
            tile,
            begin_bit=begin_bit,
            end_bit=bits,
            radix_bits=1,
            descending=True,
            memoize_outer_scan=False,
            inner_scan_algorithm="raking",
            smem_config="eight_byte",
        )
        ct.store(asc, (0,), asc_ranks)
        ct.store(desc, (0,), desc_ranks)
        ct.store(source, (0,), tile)

    ct.launch(
        torch.cuda.current_stream(),
        (1, 1, 1),
        kernel,
        (keys, ascending, descending),
    )
    torch.cuda.synchronize()
    assert torch.equal(keys, preserved)

    # Rank the sign groups directly. This oracle does not reproduce the
    # provider's bit transform, and equal signs must retain input order.
    signs = [value >= 0 for value in values * (num_items // len(values))]
    for output, reverse in ((ascending, False), (descending, True)):
        permutation = sorted(
            range(num_items), key=signs.__getitem__, reverse=reverse
        )
        expected = [0] * num_items
        for rank, index in enumerate(permutation):
            expected[index] = rank
        assert output.tolist() == expected


@pytest.mark.parametrize(
    "descending,end_bit_value", [(False, 4), (True, 4), (True, 0)]
)
def test_cuda_coop_tile_executes_radix_sort_keys(
    monkeypatch, descending, end_bit_value
):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 128
    batch_size = 2
    torch.manual_seed(23)
    keys_2d = torch.randint(
        0,
        256,
        (batch_size, num_items),
        device="cuda",
        dtype=torch.int32,
    )
    keys_in = keys_2d.flatten()
    keys_out = torch.zeros_like(keys_in)
    begin_bit_in = torch.tensor([0], device="cuda", dtype=torch.int32)
    end_bit_in = torch.tensor([end_bit_value], device="cuda", dtype=torch.int32)

    @ct.kernel
    def kernel(
        keys_arr: ct.Array,
        begin_bit_arr: ct.Array,
        end_bit_arr: ct.Array,
        keys_out_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        keys = ct.load(keys_arr, (batch_idx,), shape=(num_items,))
        begin_bit = ct.load(begin_bit_arr, (0,), shape=(1,))
        end_bit = ct.load(end_bit_arr, (0,), shape=(1,))
        sorted_keys = coop.radix_sort_keys(
            keys,
            descending=descending,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        ct.store(keys_out_arr, (batch_idx,), sorted_keys)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (keys_in, begin_bit_in, end_bit_in, keys_out),
    )
    torch.cuda.synchronize()

    out_2d = keys_out.view(batch_size, num_items)
    digits = keys_2d & ((1 << end_bit_value) - 1)
    permutation = torch.argsort(
        digits, dim=1, descending=descending, stable=True
    )
    assert torch.equal(out_2d, torch.gather(keys_2d, 1, permutation))


def test_cuda_coop_tile_executes_radix_sort_option_variants(monkeypatch):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 128
    batch_size = 2
    torch.manual_seed(41)
    keys_2d = torch.randint(
        0,
        256,
        (batch_size, num_items),
        device="cuda",
        dtype=torch.int32,
    )
    vals_2d = torch.arange(num_items, device="cuda", dtype=torch.int32)
    vals_2d = vals_2d.unsqueeze(0).repeat(batch_size, 1)
    keys_in = keys_2d.flatten()
    vals_in = vals_2d.flatten()
    keys_out = torch.zeros_like(keys_in)
    vals_out = torch.zeros_like(vals_in)

    @ct.kernel
    def kernel(
        keys_arr: ct.Array,
        vals_arr: ct.Array,
        keys_out_arr: ct.Array,
        vals_out_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        keys = ct.load(keys_arr, (batch_idx,), shape=(num_items,))
        vals = ct.load(vals_arr, (batch_idx,), shape=(num_items,))
        sorted_keys, sorted_vals = coop.radix_sort_pairs(
            keys,
            vals,
            descending=True,
            begin_bit=0,
            end_bit=8,
            radix_bits=5,
            memoize_outer_scan=False,
            inner_scan_algorithm="raking",
            smem_config="eight_byte",
        )
        ct.store(keys_out_arr, (batch_idx,), sorted_keys)
        ct.store(vals_out_arr, (batch_idx,), sorted_vals)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (keys_in, vals_in, keys_out, vals_out),
    )
    torch.cuda.synchronize()

    keys_out_2d = keys_out.view(batch_size, num_items)
    vals_out_2d = vals_out.view(batch_size, num_items).to(torch.int64)
    low_bits = keys_out_2d & 0xFF
    gathered_keys = torch.gather(keys_2d, 1, vals_out_2d)
    assert torch.all(low_bits[:, 1:] <= low_bits[:, :-1])
    assert torch.equal(gathered_keys, keys_out_2d)


@pytest.mark.parametrize(
    "dtype_name,storage_name,bits,one,infinity",
    [
        ("float16", "int16", 16, 0x3C00, 0x7C00),
        ("bfloat16", "int16", 16, 0x3F80, 0x7F80),
        ("float32", "int32", 32, 0x3F800000, 0x7F800000),
        ("float64", "int64", 64, 0x3FF0000000000000, 0x7FF0000000000000),
    ],
)
def test_radix_sort_floating_keys_preserve_bits_and_stable_pairs(
    monkeypatch, dtype_name, storage_name, bits, one, infinity
):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()
    dtype = getattr(torch, dtype_name)
    storage = getattr(torch, storage_name)
    num_items = 256
    sign = 1 << (bits - 1)
    # Ascending CUB radix order: negative NaN, -inf, -1, either zero,
    # +1, +inf, positive NaN. Both zero signs share one stable tie group.
    patterns = [
        sign | infinity | 1,
        sign | infinity,
        sign | one,
        sign,
        0,
        one,
        infinity,
        infinity | 1,
    ]
    ranks = [0, 1, 2, 3, 3, 4, 5, 6]
    pattern_ids = [
        (i * 5 + i // 8 + block * 3) % len(patterns)
        for block in range(2)
        for i in range(num_items)
    ]
    raw = [patterns[i] for i in pattern_ids]
    signed_raw = [
        value - (1 << bits) if value & sign else value for value in raw
    ]
    keys = torch.tensor(signed_raw, dtype=storage, device="cuda").view(dtype)
    values = torch.arange(2 * num_items, dtype=torch.int64, device="cuda")
    ascending = torch.empty_like(keys)
    descending = torch.empty_like(keys)
    paired_values = torch.empty_like(values)

    @ct.kernel(num_worker_warps=4)
    def kernel(
        source: ct.Array,
        ids: ct.Array,
        asc: ct.Array,
        desc: ct.Array,
        result_ids: ct.Array,
    ):
        block = ct.bid(0)
        tile = ct.load(source, (block,), shape=(num_items,))
        payload = ct.load(ids, (block,), shape=(num_items,))
        sorted_keys = coop.radix_sort_keys(tile)
        pair_keys, pair_values = coop.radix_sort_pairs(
            tile, payload, descending=True
        )
        ct.store(asc, (block,), sorted_keys)
        ct.store(desc, (block,), pair_keys)
        ct.store(result_ids, (block,), pair_values)
        ct.store(source, (block,), tile)
        ct.store(ids, (block,), payload)

    ct.launch(
        torch.cuda.current_stream(),
        (2, 1, 1),
        kernel,
        (keys, values, ascending, descending, paired_values),
    )
    torch.cuda.synchronize()
    expected_ascending = []
    expected_descending = []
    for block in range(2):
        indices = range(block * num_items, (block + 1) * num_items)
        expected_ascending.extend(
            sorted(indices, key=lambda i: ranks[pattern_ids[i]])
        )
        expected_descending.extend(
            sorted(indices, key=lambda i: ranks[pattern_ids[i]], reverse=True)
        )
    assert ascending.view(storage).tolist() == [
        signed_raw[i] for i in expected_ascending
    ]
    assert descending.view(storage).tolist() == [
        signed_raw[i] for i in expected_descending
    ]
    assert paired_values.tolist() == expected_descending
    assert keys.view(storage).tolist() == signed_raw
    assert values.tolist() == list(range(2 * num_items))


def test_cuda_coop_tile_executes_shuffle_and_radix_rank(monkeypatch):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 128
    batch_size = 2
    rotate_distance = 3
    offset_distance = 2
    begin_bit = 0
    end_bit = 4
    shuffle_2d = torch.arange(num_items, device="cuda", dtype=torch.int32)
    shuffle_2d = shuffle_2d.repeat(batch_size, 1)
    torch.manual_seed(37)
    rank_keys_2d = torch.randint(
        0,
        2**16,
        (batch_size, num_items),
        device="cuda",
        dtype=torch.int32,
    )
    shuffle_data = shuffle_2d.flatten()
    rank_keys_data = rank_keys_2d.flatten()
    up_out = torch.zeros_like(shuffle_data)
    down_out = torch.zeros_like(shuffle_data)
    offset_out = torch.zeros_like(shuffle_data)
    rotate_out = torch.zeros_like(shuffle_data)
    ranks_out = torch.zeros_like(rank_keys_data)

    @ct.kernel
    def kernel(
        shuffle_arr: ct.Array,
        rank_keys_arr: ct.Array,
        up_arr: ct.Array,
        down_arr: ct.Array,
        offset_arr: ct.Array,
        rotate_arr: ct.Array,
        ranks_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        shuffle_tile = ct.load(shuffle_arr, (batch_idx,), shape=(num_items,))
        rank_keys = ct.load(rank_keys_arr, (batch_idx,), shape=(num_items,))
        up = coop.shuffle_up(shuffle_tile)
        down = coop.shuffle_down(shuffle_tile)
        offset = coop.shuffle_offset(shuffle_tile, distance=offset_distance)
        rotated = coop.shuffle_rotate(shuffle_tile, distance=rotate_distance)
        ranks = coop.radix_rank_keys(
            rank_keys,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        ct.store(up_arr, (batch_idx,), up)
        ct.store(down_arr, (batch_idx,), down)
        ct.store(offset_arr, (batch_idx,), offset)
        ct.store(rotate_arr, (batch_idx,), rotated)
        ct.store(ranks_arr, (batch_idx,), ranks)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (
            shuffle_data,
            rank_keys_data,
            up_out,
            down_out,
            offset_out,
            rotate_out,
            ranks_out,
        ),
    )
    torch.cuda.synchronize()

    expected_up = shuffle_2d.clone()
    expected_up[:, 1:] = shuffle_2d[:, :-1]
    expected_down = shuffle_2d.clone()
    expected_down[:, :-1] = shuffle_2d[:, 1:]
    expected_rotate = torch.roll(shuffle_2d, shifts=-rotate_distance, dims=1)
    ranks_2d = ranks_out.view(batch_size, num_items)

    assert torch.equal(up_out.view_as(expected_up), expected_up)
    assert torch.equal(down_out.view_as(expected_down), expected_down)
    assert torch.equal(
        offset_out.view(batch_size, num_items)[:, :-offset_distance],
        shuffle_2d[:, offset_distance:],
    )
    assert torch.equal(rotate_out.view_as(expected_rotate), expected_rotate)
    _validate_radix_ranks(
        torch,
        rank_keys_2d,
        ranks_2d,
        begin_bit,
        end_bit,
        descending=False,
    )


@pytest.mark.parametrize("begin_bit", [0, 30])
def test_radix_rank_runtime_begin_with_default_or_runtime_end(
    monkeypatch, begin_bit
):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()
    num_items = 256
    values = [-2147483648, -1073741824, -1, 0, 1, 1073741824, 2147483647, -1]
    values *= num_items // len(values)
    keys = torch.tensor(values, device="cuda", dtype=torch.int32)
    bounds = torch.tensor(
        [begin_bit, min(32, begin_bit + 4)], device="cuda", dtype=torch.int32
    )
    default_ranks = torch.empty_like(keys)
    explicit_ranks = torch.empty_like(keys)

    @ct.kernel(num_worker_warps=4)
    def kernel(
        source: ct.Array,
        limits: ct.Array,
        implicit: ct.Array,
        explicit: ct.Array,
    ):
        tile = ct.load(source, (0,), shape=(num_items,))
        begin = ct.load(limits, (0,), shape=(1,))
        end = ct.load(limits, (1,), shape=(1,))
        first = coop.radix_rank_keys(tile, begin_bit=begin)
        second = coop.radix_rank_keys(tile, begin_bit=begin, end_bit=end)
        ct.store(implicit, (0,), first)
        ct.store(explicit, (0,), second)

    ct.launch(
        torch.cuda.current_stream(),
        (1, 1, 1),
        kernel,
        (keys, bounds, default_ranks, explicit_ranks),
    )
    torch.cuda.synchronize()
    mask = (1 << (min(32, begin_bit + 4) - begin_bit)) - 1
    digits = [((value + (1 << 31)) >> begin_bit) & mask for value in values]
    permutation = sorted(range(num_items), key=digits.__getitem__)
    expected = [0] * num_items
    for rank, index in enumerate(permutation):
        expected[index] = rank
    assert default_ranks.tolist() == expected
    assert explicit_ranks.tolist() == expected
