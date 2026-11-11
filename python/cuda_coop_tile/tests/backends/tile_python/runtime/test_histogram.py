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


@pytest.mark.parametrize(
    "num_items,bins,warps",
    [(128, 32, 1), (256, 64, 2), (256, 128, 4), (512, 256, 8)],
)
def test_cuda_coop_tile_histogram_parametric_matrix(
    monkeypatch, num_items, bins, warps
):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    batch_size = 2
    torch.manual_seed(53 + num_items + bins)
    input_2d = torch.randint(
        0,
        bins,
        (batch_size, num_items),
        device="cuda",
        dtype=torch.uint8,
    )
    input_data = input_2d.flatten()
    output_data = torch.zeros(
        batch_size * bins, device="cuda", dtype=torch.int32
    )

    original = torch.empty_like(input_data)

    @ct.kernel(num_worker_warps=warps)
    def kernel(inp: ct.Array, out: ct.Array, preserved: ct.Array):
        batch_idx = ct.bid(0)
        data = ct.load(inp, (batch_idx,), shape=(num_items,))
        counts = coop.histogram(data, bins=bins)
        ct.store(out, (batch_idx,), counts)
        ct.store(preserved, (batch_idx,), data)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (input_data, output_data, original),
    )
    torch.cuda.synchronize()

    expected = torch.stack(
        [
            torch.bincount(input_2d[batch].to(torch.int64), minlength=bins).to(
                torch.int32
            )
            for batch in range(batch_size)
        ],
        dim=0,
    )
    assert torch.equal(output_data.view_as(expected), expected)
    assert torch.equal(original, input_data)


def test_cuda_coop_tile_histogram_256(monkeypatch):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 128
    bins = 256
    batch_size = 2
    input_2d = torch.arange(
        num_items,
        device="cuda",
        dtype=torch.uint8,
    ).repeat(batch_size, 1)
    input_2d %= 17
    input_data = input_2d.flatten()
    output_data = torch.zeros(
        batch_size * bins, device="cuda", dtype=torch.int32
    )

    @ct.kernel
    def kernel(inp: ct.Array, out: ct.Array):
        batch_idx = ct.bid(0)
        data = ct.load(inp, (batch_idx,), shape=(num_items,))
        counts = coop.histogram_256(data)
        ct.store(out, (batch_idx,), counts)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (input_data, output_data),
    )
    torch.cuda.synchronize()

    expected = torch.zeros((batch_size, bins), device="cuda", dtype=torch.int32)
    for batch in range(batch_size):
        expected[batch] = torch.bincount(
            input_2d[batch].to(torch.int64), minlength=bins
        ).to(torch.int32)
    assert torch.equal(output_data.view_as(expected), expected)
