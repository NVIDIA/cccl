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


def _trace_int_field(line: str, name: str) -> int:
    prefix = f"{name}="
    return int(
        next(
            field.removeprefix(prefix)
            for field in line.split()
            if field.startswith(prefix)
        )
    )


@pytest.mark.parametrize("simt_num_warps", [4, 8])
def test_cuda_coop_tile_auto_topology_executes_varying_items_per_thread(
    monkeypatch, tmp_path, simt_num_warps
):
    trace_file = tmp_path / "provider-trace.log"
    monkeypatch.setenv("CUDA_TILE_CACHE_DIR", "off")
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE", "1")
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()
    from cuda.tile._cext import default_tile_context

    monkeypatch.setattr(default_tile_context.config, "cache_dir", None)

    num_items = 256
    block_threads = 32 * simt_num_warps
    items_per_thread = num_items // block_threads
    batch_size = 2
    input_2d = torch.arange(
        num_items,
        0,
        -1,
        device="cuda",
        dtype=torch.int32,
    ).repeat(batch_size, 1)
    input_data = input_2d.flatten()
    scan_out = torch.zeros_like(input_data)
    sum_out = torch.zeros_like(input_data)
    max_out = torch.zeros_like(input_data)
    radix_out = torch.zeros_like(input_data)

    @ct.kernel(num_worker_warps=simt_num_warps)
    def kernel(
        inp: ct.Array,
        scan_arr: ct.Array,
        sum_arr: ct.Array,
        max_arr: ct.Array,
        radix_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        values = ct.load(inp, (batch_idx,), shape=(num_items,))
        scanned = coop.exclusive_sum(values, algorithm="raking")
        summed = coop.sum(values, algorithm="raking", broadcast=True)
        maximum = coop.reduce(
            values, binary_op="max", algorithm="raking", broadcast=True
        )
        radix_sorted = coop.radix_sort_keys(
            values,
            begin_bit=0,
            end_bit=16,
            inner_scan_algorithm="raking",
        )
        ct.store(scan_arr, (batch_idx,), scanned)
        ct.store(sum_arr, (batch_idx,), summed)
        ct.store(max_arr, (batch_idx,), maximum)
        ct.store(radix_arr, (batch_idx,), radix_sorted)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (input_data, scan_out, sum_out, max_out, radix_out),
    )
    torch.cuda.synchronize()

    expected_scan = torch.cumsum(input_2d, dim=1) - input_2d
    expected_sum = input_2d.sum(dim=1, keepdim=True).repeat(1, num_items)
    expected_max = input_2d.max(dim=1, keepdim=True).values.repeat(1, num_items)
    expected_sort = torch.sort(input_2d, dim=1).values
    assert torch.equal(scan_out.view_as(expected_scan), expected_scan)
    assert torch.equal(sum_out.view_as(expected_sum), expected_sum)
    assert torch.equal(max_out.view_as(expected_max), expected_max)
    assert torch.equal(radix_out.view_as(expected_sort), expected_sort)

    resource_lines = [
        line
        for line in trace_file.read_text(encoding="utf-8").splitlines()
        if " resource_estimate " in line
    ]
    lines_by_payload = {
        next(
            field for field in line.split() if field.startswith("payload=")
        ): line
        for line in resource_lines
    }
    expected_payloads = {
        "payload=scan_sum",
        "payload=reduce_sum_batched",
        "payload=reduce_generic_batched",
        "payload=radix_sort_keys",
    }
    assert len(resource_lines) == len(expected_payloads)
    assert set(lines_by_payload) == expected_payloads
    assert {
        (
            _trace_int_field(line, "threads"),
            _trace_int_field(line, "ipt"),
        )
        for line in resource_lines
    } == {(block_threads, items_per_thread)}
    algorithm_lines = [
        lines_by_payload[payload]
        for payload in (
            "payload=scan_sum",
            "payload=reduce_sum_batched",
            "payload=reduce_generic_batched",
        )
    ]
    assert all("algorithm=raking" in line for line in algorithm_lines)
    radix_line = lines_by_payload["payload=radix_sort_keys"]
    assert "inner_scan_algorithm=raking" in radix_line


def _validate_outputs(
    torch, quantized, sorted_counts, sorted_bins, scan_counts
) -> None:
    batch_size, num_items = quantized.shape
    bins = sorted_counts.shape[1]

    counts_expected = torch.stack(
        [
            torch.bincount(quantized[batch].to(torch.int64), minlength=bins).to(
                torch.int32
            )
            for batch in range(batch_size)
        ],
        dim=0,
    )
    scan_expected = torch.zeros_like(counts_expected)
    scan_expected[:, 1:] = torch.cumsum(counts_expected[:, :-1], dim=1)

    counts_from_scan = torch.empty_like(scan_counts)
    counts_from_scan[:, :-1] = scan_counts[:, 1:] - scan_counts[:, :-1]
    counts_from_scan[:, -1] = num_items - scan_counts[:, -1]
    gathered = torch.gather(counts_from_scan, 1, sorted_bins.to(torch.int64))

    assert torch.equal(scan_counts, scan_expected)
    assert torch.all(sorted_counts[:, 1:] >= sorted_counts[:, :-1])
    assert torch.equal(gathered, sorted_counts)


def test_cuda_coop_tile_executes_histogram_sort_and_scan_with_cudatile(
    monkeypatch,
):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    bins = 256
    batch_size, m, n = 2, 16, 16
    torch.manual_seed(11)
    base = torch.randn(batch_size, m, n, device="cuda", dtype=torch.float32)
    y_in = torch.sign(base) * torch.exp(torch.abs(base) * 0.5)
    sorted_counts_out = torch.zeros(
        batch_size * bins, device="cuda", dtype=torch.int32
    )
    sorted_bins_out = torch.zeros(
        batch_size * bins, device="cuda", dtype=torch.int32
    )
    scan_out = torch.zeros(batch_size * bins, device="cuda", dtype=torch.int32)
    quantized_out = torch.empty(
        batch_size * m * n, device="cuda", dtype=torch.uint8
    )

    @ct.kernel
    def kernel(
        y_in_arr: ct.Array,
        counts_arr: ct.Array,
        bins_arr: ct.Array,
        scan_arr: ct.Array,
        quantized_arr: ct.Array,
    ):
        bid = ct.bid(0)
        y = ct.load(y_in_arr, (bid, 0, 0), shape=(1, m, n))
        y = ct.reshape(y, (m, n))
        mag = ct.maximum(y, -y)
        logv = ct.log(mag + 1.0)
        lo = ct.min(logv)
        hi = ct.max(logv)
        scaled = ct.maximum(
            ct.minimum((logv - lo) / (hi - lo + 1e-6), 1.0), 0.0
        )
        q = ct.floor(scaled * (bins - 1))
        keys = ct.reshape(ct.astype(q, ct.uint8), (m * n,))

        counts = coop.histogram(keys, bins=bins)
        bin_ids = ct.arange(bins, dtype=ct.int32)
        sorted_counts, sorted_bins = coop.radix_sort_pairs(
            counts,
            bin_ids,
            begin_bit=0,
            end_bit=32,
        )
        scan_counts = coop.exclusive_sum(counts)

        ct.store(counts_arr, (bid,), sorted_counts)
        ct.store(bins_arr, (bid,), sorted_bins)
        ct.store(scan_arr, (bid,), scan_counts)
        # Reference the exact device quantization; host log rounding can
        # otherwise move values across integer bin boundaries.
        ct.store(quantized_arr, (bid,), keys)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (y_in, sorted_counts_out, sorted_bins_out, scan_out, quantized_out),
    )
    torch.cuda.synchronize()

    _validate_outputs(
        torch,
        quantized_out.view(batch_size, m * n),
        sorted_counts_out.view(batch_size, bins),
        sorted_bins_out.view(batch_size, bins),
        scan_out.view(batch_size, bins),
    )


def test_cuda_coop_tile_executes_scan_variants_and_histogram_256(monkeypatch):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 128
    bins = 256
    batch_size = 2
    exclusive_prefix = 3.5
    inclusive_prefix = 2.0
    initial_value = 10_000
    float_2d = torch.arange(
        1,
        num_items + 1,
        device="cuda",
        dtype=torch.float32,
    ).repeat(batch_size, 1)
    torch.manual_seed(19)
    scan_2d = torch.randint(
        0,
        256,
        (batch_size, num_items),
        device="cuda",
        dtype=torch.int32,
    )
    hist_2d = torch.randint(
        0,
        bins,
        (batch_size, num_items),
        device="cuda",
        dtype=torch.uint8,
    )
    float_data = float_2d.flatten()
    scan_data = scan_2d.flatten()
    hist_data = hist_2d.flatten()
    exclusive_out = torch.zeros_like(float_data)
    inclusive_out = torch.zeros_like(float_data)
    default_exclusive_scan_out = torch.zeros_like(scan_data)
    default_scan_out = torch.zeros_like(scan_data)
    scan_out = torch.zeros_like(scan_data)
    hist_out = torch.zeros(batch_size * bins, device="cuda", dtype=torch.int32)

    @ct.kernel
    def kernel(
        float_in: ct.Array,
        scan_in: ct.Array,
        hist_in: ct.Array,
        exclusive_arr: ct.Array,
        inclusive_arr: ct.Array,
        default_exclusive_scan_arr: ct.Array,
        default_scan_arr: ct.Array,
        scan_arr: ct.Array,
        hist_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        float_tile = ct.load(float_in, (batch_idx,), shape=(num_items,))
        scan_tile = ct.load(scan_in, (batch_idx,), shape=(num_items,))
        hist_tile = ct.load(hist_in, (batch_idx,), shape=(num_items,))
        exclusive = coop.exclusive_sum(
            float_tile, prefix_value=exclusive_prefix
        )
        inclusive = coop.inclusive_sum(
            float_tile, prefix_value=inclusive_prefix
        )
        default_exclusive_scan = coop.exclusive_scan(scan_tile)
        default_scan = coop.scan(scan_tile)
        min_scan = coop.exclusive_scan(
            scan_tile,
            scan_op="min",
            initial_value=initial_value,
        )
        counts = coop.histogram_256(hist_tile)
        ct.store(exclusive_arr, (batch_idx,), exclusive)
        ct.store(inclusive_arr, (batch_idx,), inclusive)
        ct.store(
            default_exclusive_scan_arr,
            (batch_idx,),
            default_exclusive_scan,
        )
        ct.store(default_scan_arr, (batch_idx,), default_scan)
        ct.store(scan_arr, (batch_idx,), min_scan)
        ct.store(hist_arr, (batch_idx,), counts)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (
            float_data,
            scan_data,
            hist_data,
            exclusive_out,
            inclusive_out,
            default_exclusive_scan_out,
            default_scan_out,
            scan_out,
            hist_out,
        ),
    )
    torch.cuda.synchronize()

    expected_exclusive = torch.cumsum(float_2d, dim=1) - float_2d
    expected_exclusive = expected_exclusive + exclusive_prefix
    expected_inclusive = torch.cumsum(float_2d, dim=1) + inclusive_prefix
    expected_default_scan = torch.cumsum(scan_2d, dim=1) - scan_2d
    expected_scan = torch.empty_like(scan_2d)
    expected_scan[:, 0] = initial_value
    running = torch.full(
        (batch_size,),
        initial_value,
        device="cuda",
        dtype=torch.int32,
    )
    for i in range(1, num_items):
        running = torch.minimum(running, scan_2d[:, i - 1])
        expected_scan[:, i] = running
    expected_hist = torch.stack(
        [
            torch.bincount(hist_2d[batch].to(torch.int64), minlength=bins).to(
                torch.int32
            )
            for batch in range(batch_size)
        ],
        dim=0,
    )

    assert torch.allclose(
        exclusive_out.view_as(expected_exclusive),
        expected_exclusive,
    )
    assert torch.allclose(
        inclusive_out.view_as(expected_inclusive),
        expected_inclusive,
    )
    assert torch.equal(
        default_exclusive_scan_out.view_as(expected_default_scan),
        expected_default_scan,
    )
    assert torch.equal(
        default_scan_out.view_as(expected_default_scan),
        expected_default_scan,
    )
    assert torch.equal(scan_out.view_as(expected_scan), expected_scan)
    assert torch.equal(hist_out.view_as(expected_hist), expected_hist)


def test_cuda_coop_tile_executes_generic_scan_and_reduce_ops(monkeypatch):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    num_items = 128
    batch_size = 2
    torch.manual_seed(61)
    scan_2d = torch.randint(
        0,
        256,
        (batch_size, num_items),
        device="cuda",
        dtype=torch.int32,
    )
    reduce_2d = torch.rand(
        batch_size,
        num_items,
        device="cuda",
        dtype=torch.float32,
    )
    reduce_2d = reduce_2d * 0.5 + 0.75
    scan_data = scan_2d.flatten()
    reduce_data = reduce_2d.flatten()
    generic_scan_out = torch.zeros_like(scan_data)
    inclusive_scan_out = torch.zeros_like(scan_data)
    reduce_out = torch.zeros_like(reduce_data)

    @ct.kernel
    def kernel(
        scan_in: ct.Array,
        reduce_in: ct.Array,
        generic_scan_arr: ct.Array,
        inclusive_scan_arr: ct.Array,
        reduce_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        scan_tile = ct.load(scan_in, (batch_idx,), shape=(num_items,))
        reduce_tile = ct.load(reduce_in, (batch_idx,), shape=(num_items,))
        generic_scan = coop.scan(scan_tile, mode="inclusive", scan_op="max")
        inclusive_scan = coop.inclusive_scan(scan_tile, scan_op="max")
        product = coop.reduce(reduce_tile, binary_op="*", broadcast=True)
        ct.store(generic_scan_arr, (batch_idx,), generic_scan)
        ct.store(inclusive_scan_arr, (batch_idx,), inclusive_scan)
        ct.store(reduce_arr, (batch_idx,), product)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (
            scan_data,
            reduce_data,
            generic_scan_out,
            inclusive_scan_out,
            reduce_out,
        ),
    )
    torch.cuda.synchronize()

    expected_scan = torch.cummax(scan_2d, dim=1).values
    expected_reduce = reduce_2d.prod(dim=1, keepdim=True).repeat(1, num_items)

    assert torch.equal(generic_scan_out.view_as(expected_scan), expected_scan)
    assert torch.equal(inclusive_scan_out.view_as(expected_scan), expected_scan)
    assert torch.allclose(
        reduce_out.view_as(expected_reduce),
        expected_reduce,
        rtol=1e-5,
        atol=1e-5,
    )
