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


def test_cuda_coop_tile_executes_reduce_sum_and_min_with_cudatile(monkeypatch):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 128
    batch_size = 2
    torch_rng = torch.Generator(device="cuda").manual_seed(29)
    float_2d = torch.arange(
        1,
        num_items + 1,
        device="cuda",
        dtype=torch.float32,
    ).repeat(batch_size, 1)
    int_2d = torch.randint(
        -100,
        100,
        (batch_size, num_items),
        device="cuda",
        dtype=torch.int32,
        generator=torch_rng,
    )
    float_data = float_2d.flatten()
    int_data = int_2d.flatten()
    sum_out = torch.zeros_like(float_data)
    min_out = torch.zeros_like(int_data)

    @ct.kernel
    def kernel(
        float_in: ct.Array,
        sum_arr: ct.Array,
        int_in: ct.Array,
        min_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        float_tile = ct.load(float_in, (batch_idx,), shape=(num_items,))
        int_tile = ct.load(int_in, (batch_idx,), shape=(num_items,))
        ct.store(
            sum_arr,
            (batch_idx,),
            coop.reduce(float_tile, binary_op="sum", broadcast=True),
        )
        ct.store(
            min_arr,
            (batch_idx,),
            coop.reduce(int_tile, binary_op="min", broadcast=True),
        )

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (float_data, sum_out, int_data, min_out),
    )
    torch.cuda.synchronize()

    expected_sum = float_2d.sum(dim=1, keepdim=True).repeat(1, num_items)
    expected_min = int_2d.min(dim=1, keepdim=True).values.repeat(1, num_items)
    assert torch.allclose(sum_out.view_as(expected_sum), expected_sum)
    assert torch.equal(min_out.view_as(expected_min), expected_min)


@pytest.mark.parametrize("num_items", [128, 256, 512])
def test_cuda_coop_tile_sum_items_per_thread_matrix(monkeypatch, num_items):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    batch_size = 2
    input_2d = torch.arange(
        1,
        num_items + 1,
        device="cuda",
        dtype=torch.float32,
    ).repeat(batch_size, 1)
    input_data = input_2d.flatten()
    output_data = torch.zeros_like(input_data)

    @ct.kernel
    def kernel(inp: ct.Array, out: ct.Array):
        batch_idx = ct.bid(0)
        data = ct.load(inp, (batch_idx,), shape=(num_items,))
        result = coop.sum(data, broadcast=True)
        ct.store(out, (batch_idx,), result)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (input_data, output_data),
    )
    torch.cuda.synchronize()

    expected = input_2d.sum(dim=1, keepdim=True).repeat(1, num_items)
    assert torch.allclose(output_data.view_as(expected), expected)


@pytest.mark.parametrize(
    "worker_warps,dtype_name",
    [
        (None, "float64"),
        (4, "float64"),
        (8, "float64"),
        (16, "float64"),
        (8, "float32"),
        (16, "int32"),
    ],
)
def test_axis_reductions_compose_across_worker_counts(
    monkeypatch, worker_warps, dtype_name
):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()
    dtype = getattr(torch, dtype_name)
    host = (torch.arange(1024).reshape(8, 128) * 17 % 101 - 50).to(dtype)
    if dtype.is_floating_point:
        host = host / 7
    source = host.cuda()
    vector = source[0].clone()
    rows = torch.empty((8,), dtype=dtype, device="cuda")
    kept = torch.empty((8, 1), dtype=dtype, device="cuda")
    minima = torch.empty_like(rows)
    maxima = torch.empty_like(rows)
    total = torch.empty((1,), dtype=dtype, device="cuda")
    dependent = torch.empty_like(total)
    kernel_options = (
        {} if worker_warps is None else {"num_worker_warps": worker_warps}
    )

    @ct.kernel(**kernel_options)
    def kernel(source, vector, rows, kept, minima, maxima, total, dependent):
        values = ct.load(source, (0, 0), (8, 128))
        lanes = ct.load(vector, (0,), (128,))
        scalar = coop.sum(lanes)
        row_sums = coop.sum(values, axis=1)
        retained = coop.sum(values, axis=-1, keepdims=True)
        minimum = coop.reduce(values, axis=1, binary_op="min")
        maximum = coop.reduce(values, axis=1, binary_op="max")
        combined = coop.sum(row_sums)
        ct.static_assert(scalar.shape == ())
        ct.static_assert(row_sums.shape == (8,))
        ct.static_assert(retained.shape == (8, 1))
        ct.static_assert(combined.shape == ())
        ct.store(rows, (0,), row_sums)
        ct.store(kept, (0, 0), retained)
        ct.store(minima, (0,), minimum)
        ct.store(maxima, (0,), maximum)
        ct.store(total, (0,), ct.reshape(scalar, (1,)))
        ct.store(dependent, (0,), ct.reshape(combined, (1,)))
        ct.store(source, (0, 0), values)
        ct.store(vector, (0,), lanes)

    ct.launch(
        torch.cuda.current_stream(),
        (1,),
        kernel,
        (source, vector, rows, kept, minima, maxima, total, dependent),
    )
    torch.cuda.synchronize()
    tolerance = {"rtol": 2e-5, "atol": 2e-4} if dtype == torch.float32 else {}
    reference = host.sum(dim=1, dtype=dtype)
    torch.testing.assert_close(rows.cpu(), reference, **tolerance)
    torch.testing.assert_close(kept.cpu(), reference[:, None], **tolerance)
    torch.testing.assert_close(minima.cpu(), host.amin(dim=1))
    torch.testing.assert_close(maxima.cpu(), host.amax(dim=1))
    torch.testing.assert_close(
        total.cpu()[0], host[0].sum(dtype=dtype), **tolerance
    )
    torch.testing.assert_close(
        dependent.cpu()[0], host.sum(dtype=dtype), **tolerance
    )
    assert torch.equal(source.cpu(), host)
    assert torch.equal(vector.cpu(), host[0])


def test_non_last_tuple_axes_and_broadcast_result(monkeypatch):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()
    host = (torch.arange(512).reshape(4, 8, 16) * 13 % 89 - 44).to(torch.int32)
    source = host.cuda()
    sums = torch.empty((8,), dtype=torch.int32, device="cuda")
    kept = torch.empty((1, 8, 1), dtype=torch.int32, device="cuda")
    minima = torch.empty((8, 16), dtype=torch.int32, device="cuda")
    repeated = torch.empty_like(source)
    unchanged = torch.empty_like(source)

    @ct.kernel(num_worker_warps=16)
    def kernel(source, sums, kept, minima, repeated, unchanged):
        values = ct.load(source, (0, 0, 0), (4, 8, 16))
        total = coop.sum(values, axis=(2, 0))
        retained = coop.reduce(
            values, axis=(-1, -3), keepdims=True, binary_op="sum"
        )
        minimum = coop.reduce(values, axis=0, binary_op="min")
        broadcast = coop.sum(values, axis=(0, 2), broadcast=True)
        identity = coop.sum(values, axis=())
        ct.static_assert(total.shape == (8,))
        ct.static_assert(retained.shape == (1, 8, 1))
        ct.static_assert(minimum.shape == (8, 16))
        ct.static_assert(broadcast.shape == (4, 8, 16))
        ct.static_assert(identity.shape == values.shape)
        ct.store(sums, (0,), total)
        ct.store(kept, (0, 0, 0), retained)
        ct.store(minima, (0, 0), minimum)
        ct.store(repeated, (0, 0, 0), broadcast)
        ct.store(unchanged, (0, 0, 0), identity)

    ct.launch(
        torch.cuda.current_stream(),
        (1,),
        kernel,
        (source, sums, kept, minima, repeated, unchanged),
    )
    torch.cuda.synchronize()
    reference = host.sum(dim=(0, 2), dtype=torch.int32)
    assert torch.equal(sums.cpu(), reference)
    assert torch.equal(kept.cpu(), reference[None, :, None])
    assert torch.equal(minima.cpu(), host.amin(dim=0))
    assert torch.equal(repeated.cpu(), reference[None, :, None].expand_as(host))
    assert torch.equal(unchanged.cpu(), host)


@pytest.mark.parametrize("algorithm", ["warp_reductions", "raking"])
def test_partial_reductions_preserve_nonfinite_values(monkeypatch, algorithm):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()
    host = torch.tensor(
        [
            [float("nan")] * 8,
            [float("inf")] * 8,
            [-float("inf")] * 8,
            [float("inf"), -float("inf"), 1, 2, 3, 4, 5, 6],
            [-0.0] * 8,
            [0.0] * 8,
            [-8, -4, -2, -1, 1, 2, 4, 8],
            [1e-310] * 8,
        ],
        dtype=torch.float64,
    )
    source = host.cuda()
    sums = torch.empty((8,), dtype=source.dtype, device="cuda")
    minima = torch.empty_like(sums)
    maxima = torch.empty_like(sums)

    @ct.kernel(num_worker_warps=16)
    def kernel(source, sums, minima, maxima):
        values = ct.load(source, (0, 0), (8, 8))
        ct.store(sums, (0,), coop.sum(values, axis=1, algorithm=algorithm))
        ct.store(
            minima,
            (0,),
            coop.reduce(values, axis=1, binary_op="min", algorithm=algorithm),
        )
        ct.store(
            maxima,
            (0,),
            coop.reduce(values, axis=1, binary_op="max", algorithm=algorithm),
        )
        ct.store(source, (0, 0), values)

    ct.launch(
        torch.cuda.current_stream(),
        (1,),
        kernel,
        (source, sums, minima, maxima),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(sums.cpu(), host.sum(dim=1), equal_nan=True)
    torch.testing.assert_close(minima.cpu(), host.amin(dim=1), equal_nan=True)
    torch.testing.assert_close(maxima.cpu(), host.amax(dim=1), equal_nan=True)
    # Tie signs follow the existing generic operator. Sum's zero sign can
    # differ with the reduction order and is deliberately not a parity claim.
    assert torch.signbit(minima.cpu()[4])
    assert torch.signbit(maxima.cpu()[4])
    assert sums.cpu()[7] > 0
    assert torch.equal(source.cpu().view(torch.int64), host.view(torch.int64))


@pytest.mark.parametrize("row_size", [2, 4])
def test_boolean_sum_promotes_to_int32(monkeypatch, row_size):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()
    source = (
        torch.arange(8 * row_size, device="cuda").reshape(8, row_size) % 3
    ) == 0
    total = torch.empty((1,), dtype=torch.int32, device="cuda")
    rows = torch.empty((8,), dtype=torch.int32, device="cuda")
    unchanged = torch.empty((8, row_size), dtype=torch.int32, device="cuda")

    @ct.kernel(num_worker_warps=8)
    def kernel(source, total, rows, unchanged):
        values = ct.load(source, (0, 0), (8, row_size))
        scalar = coop.sum(values)
        counts = coop.reduce(values, axis=1, binary_op="sum")
        identity = coop.sum(values, axis=())
        ct.static_assert(scalar.shape == ())
        ct.static_assert(scalar.dtype == ct.int32)
        ct.static_assert(identity.dtype == ct.int32)
        ct.store(total, (0,), ct.reshape(scalar, (1,)))
        ct.store(rows, (0,), counts)
        ct.store(unchanged, (0, 0), identity)

    ct.launch(
        torch.cuda.current_stream(),
        (1,),
        kernel,
        (source, total, rows, unchanged),
    )
    torch.cuda.synchronize()
    assert torch.equal(total.cpu()[0], source.cpu().sum(dtype=torch.int32))
    assert torch.equal(rows.cpu(), source.cpu().sum(dim=1, dtype=torch.int32))
    assert torch.equal(unchanged.cpu(), source.cpu().to(torch.int32))
