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


def test_scan_reuses_supported_worker_width_for_distinct_extents(
    monkeypatch, tmp_path
):
    trace_file = tmp_path / "provider-trace.log"
    compile_counter = tmp_path / "provider-compiles.log"
    monkeypatch.setenv("CUDA_TILE_CACHE_DIR", "off")
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE", "1")
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))
    monkeypatch.setenv(
        "CUDA_COOP_TILE_PROVIDER_NVRTC_COMPILE_COUNTER_FILE",
        str(compile_counter),
    )
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()
    from cuda.tile._cext import default_tile_context

    # The default context may have been created by an earlier runtime test,
    # before CUDA_TILE_CACHE_DIR was changed above. Disable its live cache too
    # so this test always exercises discovery and NVRTC exactly once.
    monkeypatch.setattr(default_tile_context.config, "cache_dir", None)

    batch_size = 2
    small_items = 128
    large_items = 512
    small_2d = torch.arange(
        1,
        small_items + 1,
        device="cuda",
        dtype=torch.int32,
    ).repeat(batch_size, 1)
    large_2d = torch.arange(
        1,
        large_items + 1,
        device="cuda",
        dtype=torch.int32,
    ).repeat(batch_size, 1)
    small_2d[1] += 3
    large_2d[1] += 7
    small_data = small_2d.flatten()
    large_data = large_2d.flatten()
    small_out = torch.zeros_like(small_data)
    large_out = torch.zeros_like(large_data)

    @ct.kernel
    def kernel(
        small_in: ct.Array,
        large_in: ct.Array,
        small_out_arr: ct.Array,
        large_out_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        small = ct.load(small_in, (batch_idx,), shape=(small_items,))
        large = ct.load(large_in, (batch_idx,), shape=(large_items,))

        # Both extents admit the preview's supported 128-worker width.
        # Their distinct per-thread extents and scratch remain compiler-visible.
        small_scan = coop.exclusive_scan(small, scan_op="sum")
        large_scan = coop.exclusive_sum(large)

        ct.store(small_out_arr, (batch_idx,), small_scan)
        ct.store(large_out_arr, (batch_idx,), large_scan)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size, 1, 1),
        kernel,
        (small_data, large_data, small_out, large_out),
    )
    torch.cuda.synchronize()

    expected_small = torch.cumsum(small_2d, dim=1) - small_2d
    expected_large = torch.cumsum(large_2d, dim=1) - large_2d
    assert torch.equal(small_out.view_as(expected_small), expected_small)
    assert torch.equal(large_out.view_as(expected_large), expected_large)

    trace_lines = trace_file.read_text(encoding="utf-8").splitlines()
    resource_lines = [
        line for line in trace_lines if " resource_estimate " in line
    ]
    candidate_lines = [
        line for line in trace_lines if " discovery candidate=" in line
    ]
    assert candidate_lines
    candidate_scores = {
        (
            _trace_int_field(line, "threads"),
            _trace_int_field(line, "ipt"),
        ): _trace_int_field(line, "provider_score")
        for line in candidate_lines
    }
    selected_topologies = {
        (
            _trace_int_field(line, "threads"),
            _trace_int_field(line, "ipt"),
        )
        for line in resource_lines
    }
    assert {threads * items for threads, items in selected_topologies} == {
        small_items,
        large_items,
    }
    assert selected_topologies == {(128, 1), (128, 4)}
    # Catalog preferences describe each candidate before the compiler chooses
    # its worker width. They do not force that whole-kernel choice.
    for extent in (small_items, large_items):
        scores = {
            topology: score
            for topology, score in candidate_scores.items()
            if topology[0] * topology[1] == extent
        }
        assert len(set(scores.values())) > 1
        assert max(scores, key=scores.get) == (128, extent // 128)
    assert compile_counter.read_text(encoding="utf-8").splitlines() == ["1"]


@pytest.mark.parametrize("extent", [128, 512])
def test_cuda_coop_tile_scan_prefixes_match_independent_host_fold(
    monkeypatch, extent
):
    _configure_e2e_environment(monkeypatch)
    torch, _, _ = _import_e2e_modules()

    # Match the C++ scan-prefix consumer's inputs and eight public operations.
    # The expected values use a sequential host fold, independent of either DSL.
    @ct.kernel(num_worker_warps=4)
    def kernel(source: ct.Array, seed_input: ct.Array, output: ct.Array):
        values = ct.load(source, (0,), shape=(extent,))
        seed = ct.load(seed_input, (0,), shape=(1,))
        ct.store(output, (0,), coop.exclusive_sum(values, prefix_value=seed))
        ct.store(
            output,
            (1,),
            coop.inclusive_sum(
                values, prefix_value=seed, algorithm="raking_memoize"
            ),
        )
        ct.store(
            output,
            (2,),
            coop.inclusive_scan(values, scan_op="max", prefix_value=seed),
        )
        ct.store(output, (3,), coop.inclusive_scan(values, prefix_value=seed))
        ct.store(
            output,
            (4,),
            coop.scan(values, mode="inclusive", scan_op="^", prefix_value=seed),
        )
        ct.store(
            output,
            (5,),
            coop.scan(
                values,
                mode="inclusive",
                prefix_value=seed,
                algorithm="warp_scans",
            ),
        )
        ct.store(
            output,
            (6,),
            coop.scan(values, scan_op="max", prefix_value=seed),
        )
        ct.store(output, (7,), coop.scan(values, prefix_value=seed))
        ct.store(source, (0,), values)

    host_input = [(index * 7) % 11 - 5 for index in range(extent)]
    source = torch.tensor(host_input, dtype=torch.int32, device="cuda")
    preserved = source.clone()
    for prefix in (-7, 19):
        seed_input = torch.tensor([prefix], dtype=torch.int32, device="cuda")
        output = torch.empty(8 * extent, dtype=torch.int32, device="cuda")
        ct.launch(
            torch.cuda.current_stream(),
            (1, 1, 1),
            kernel,
            (source, seed_input, output),
        )
        torch.cuda.synchronize()
        total = maximum = bits = prefix
        expected = [[] for _ in range(8)]
        for value in host_input:
            previous_total, previous_maximum = total, maximum
            total += value
            maximum = max(maximum, value)
            bits ^= value
            for row, result in zip(
                expected,
                (
                    previous_total,
                    total,
                    maximum,
                    total,
                    bits,
                    total,
                    previous_maximum,
                    previous_total,
                ),
            ):
                row.append(result)
        assert output.cpu().reshape(8, extent).tolist() == expected
        assert torch.equal(source, preserved)
