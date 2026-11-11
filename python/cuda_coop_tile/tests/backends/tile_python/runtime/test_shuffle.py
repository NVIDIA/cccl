# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.e2e import configure_e2e_environment, import_e2e_modules


@pytest.mark.parametrize("offset", [-7, 9])
def test_shuffle_constant_and_runtime_distances(monkeypatch, offset):
    configure_e2e_environment(monkeypatch)
    global ct, coop
    torch, ct, coop = import_e2e_modules()
    num_items = 128
    values = torch.arange(num_items, device="cuda", dtype=torch.int32) * 3 + 1
    outputs = [torch.empty_like(values) for _ in range(5)]

    @ct.kernel(num_worker_warps=4)
    def kernel(inp, dynamic_offset, dynamic_rotation, a, b, c, d, original):
        tile = ct.load(inp, (0,), shape=(num_items,))
        ct.store(a, (0,), coop.shuffle_offset(tile, distance=dynamic_offset))
        ct.store(b, (0,), coop.shuffle_rotate(tile, distance=dynamic_rotation))
        ct.store(c, (0,), coop.shuffle_offset(tile, distance=-3))
        ct.store(d, (0,), coop.shuffle_rotate(tile, distance=5))
        ct.store(original, (0,), tile)

    ct.launch(
        torch.cuda.current_stream(), (1,), kernel, (values, offset, 7, *outputs)
    )
    torch.cuda.synchronize()
    indices = torch.arange(num_items, device="cuda")
    for actual, distance in ((outputs[0], offset), (outputs[2], -3)):
        source = indices + distance
        expected = torch.where(
            (source >= 0) & (source < num_items),
            values[source.clamp(0, num_items - 1)],
            values,
        )
        assert torch.equal(actual, expected)
    assert torch.equal(outputs[1], values[(indices + 7) % num_items])
    assert torch.equal(outputs[3], values[(indices + 5) % num_items])
    assert torch.equal(outputs[4], values)


@pytest.mark.parametrize(
    ("mode", "distance", "diagnostic"),
    [
        ("offset", -128, "distance must satisfy"),
        ("offset", 128, "distance must satisfy"),
        ("rotate", -1, "distance must be >= 0"),
        ("rotate", 128, "distance must be < tile size"),
    ],
)
def test_shuffle_rejects_invalid_constant_distance(
    monkeypatch, mode, distance, diagnostic
):
    configure_e2e_environment(monkeypatch)
    global ct, coop
    torch, ct, coop = import_e2e_modules()
    from cuda.tile._exception import TileTypeError

    values = torch.arange(128, device="cuda", dtype=torch.int32)
    output = torch.empty_like(values)

    @ct.kernel(num_worker_warps=4)
    def kernel(inp, out):
        tile = ct.load(inp, (0,), shape=(128,))
        if mode == "offset":
            result = coop.shuffle_offset(tile, distance=distance)
        else:
            result = coop.shuffle_rotate(tile, distance=distance)
        ct.store(out, (0,), result)

    with pytest.raises(TileTypeError, match=diagnostic):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (values, output))
