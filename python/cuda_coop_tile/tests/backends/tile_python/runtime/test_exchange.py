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


def test_cuda_coop_tile_executes_exchange_roundtrip_with_cudatile(monkeypatch):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 256
    batch_size = 2
    input_2d = torch.arange(num_items, device="cuda", dtype=torch.int32)
    input_2d = (input_2d * 7 + 3).repeat(batch_size, 1)
    input_data = input_2d.flatten()
    output_data = torch.zeros_like(input_data)

    @ct.kernel
    def kernel(inp: ct.Array, out: ct.Array):
        batch_idx = ct.bid(0)
        samples = ct.load(inp, (batch_idx,), shape=(num_items,))
        blocked = coop.exchange_striped_to_blocked(samples, block_threads=128)
        striped = coop.exchange_blocked_to_striped(blocked, block_threads=128)
        ct.store(out, (batch_idx,), striped)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (input_data, output_data),
    )
    torch.cuda.synchronize()

    assert torch.equal(output_data.view(batch_size, num_items), input_2d)


@pytest.mark.parametrize("block_threads", [128, 256])
def test_cuda_coop_tile_executes_exchange_directional_layouts(
    monkeypatch, block_threads
):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 256
    batch_size = 2
    input_2d = torch.arange(num_items, device="cuda", dtype=torch.int32)
    input_2d = input_2d.repeat(batch_size, 1)
    input_data = input_2d.flatten()
    striped_to_blocked_out = torch.zeros_like(input_data)
    blocked_to_striped_out = torch.zeros_like(input_data)
    original_out = torch.zeros_like(input_data)

    @ct.kernel(num_worker_warps=block_threads // 32)
    def kernel(
        inp: ct.Array,
        s2b_out: ct.Array,
        b2s_out: ct.Array,
        preserved_out: ct.Array,
    ):
        batch_idx = ct.bid(0)
        samples = ct.load(inp, (batch_idx,), shape=(num_items,))
        striped_to_blocked = coop.exchange_striped_to_blocked(
            samples, block_threads=block_threads
        )
        blocked_to_striped = coop.exchange_blocked_to_striped(
            samples, block_threads=block_threads
        )
        ct.store(s2b_out, (batch_idx,), striped_to_blocked)
        ct.store(b2s_out, (batch_idx,), blocked_to_striped)
        ct.store(preserved_out, (batch_idx,), samples)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (
            input_data,
            striped_to_blocked_out,
            blocked_to_striped_out,
            original_out,
        ),
    )
    torch.cuda.synchronize()

    items_per_thread = num_items // block_threads
    expected_s2b = (
        input_2d.reshape(batch_size, items_per_thread, block_threads)
        .permute(0, 2, 1)
        .reshape(batch_size, num_items)
    )
    expected_b2s = (
        input_2d.reshape(batch_size, block_threads, items_per_thread)
        .permute(0, 2, 1)
        .reshape(batch_size, num_items)
    )

    assert torch.equal(
        striped_to_blocked_out.view_as(expected_s2b), expected_s2b
    )
    assert torch.equal(
        blocked_to_striped_out.view_as(expected_b2s), expected_b2s
    )
    assert torch.equal(original_out.view_as(input_2d), input_2d)
