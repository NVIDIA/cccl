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


@pytest.mark.parametrize("simt_num_warps", [1, 2])
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
