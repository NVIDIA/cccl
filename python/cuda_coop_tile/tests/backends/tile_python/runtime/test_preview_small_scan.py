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


class _AutomaticWorkerSelectionFailure(Exception):
    pass


# The CUDA 13.5 compiler defaults to 128 workers for these
# mixed 64/512-item CTA scans, rejecting the 64-item call. Two worker warps
# provide a workaround. Keep the automatic case enabled to detect compiler
# worker-selection changes.
@pytest.mark.parametrize(
    "worker_warps", [None, 2], ids=["automatic", "two-warps"]
)
def test_cuda_coop_tile_block_scan_resolves_mixed_extents(
    monkeypatch, tmp_path, worker_warps, request
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
    from cuda.tile._compile import _get_compiler_version_string
    from cuda.tile._exception import CompilerExecutionError

    compiler_version = _get_compiler_version_string() or ""
    known_failure = worker_warps is None and "release 13.5," in compiler_version
    request.node.add_marker(
        pytest.mark.xfail(
            condition=known_failure,
            raises=_AutomaticWorkerSelectionFailure,
            strict=True,
            reason="CUDA 13.5 automatic workers reject mixed 64/512-item Scan",
        )
    )

    # The default context may have been created by an earlier runtime test,
    # before CUDA_TILE_CACHE_DIR was changed above. Disable its live cache too
    # so this test always exercises discovery and NVRTC exactly once.
    monkeypatch.setattr(default_tile_context.config, "cache_dir", None)

    batch_size = 2
    small_items = 64
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

    @ct.kernel(num_worker_warps=worker_warps)
    def kernel(
        small_in: ct.Array,
        large_in: ct.Array,
        small_out_arr: ct.Array,
        large_out_arr: ct.Array,
    ):
        batch_idx = ct.bid(0)
        small = ct.load(small_in, (batch_idx,), shape=(small_items,))
        large = ct.load(large_in, (batch_idx,), shape=(large_items,))

        # These calls have different frontend validation topologies (64 and
        # 128 threads), but both admit one compiler-resolved CTA width. Their
        # scratch requirements and synchronization remain compiler-visible.
        small_scan = coop.exclusive_sum(small)
        large_scan = coop.exclusive_sum(large)

        ct.store(small_out_arr, (batch_idx,), small_scan)
        ct.store(large_out_arr, (batch_idx,), large_scan)

    try:
        ct.launch(
            torch.cuda.current_stream(),
            (batch_size, 1, 1),
            kernel,
            (small_data, large_data, small_out, large_out),
        )
    except CompilerExecutionError as error:
        diagnostic = [
            line
            for line in error.message.splitlines()
            if line and not line.startswith("[coop_linkage_provider][trace]")
        ]
        if known_failure and diagnostic == [
            "Return code 5",
            "failed to compile Tile IR program",
        ]:
            raise _AutomaticWorkerSelectionFailure(str(error)) from error
        raise
    torch.cuda.synchronize()

    expected_small = torch.cumsum(small_2d, dim=1) - small_2d
    expected_large = torch.cumsum(large_2d, dim=1) - large_2d
    assert torch.equal(small_out.view_as(expected_small), expected_small)
    assert torch.equal(large_out.view_as(expected_large), expected_large)

    resource_lines = [
        line
        for line in trace_file.read_text(encoding="utf-8").splitlines()
        if " resource_estimate " in line
    ]
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
    assert len({threads for threads, _ in selected_topologies}) == 1
    assert compile_counter.read_text(encoding="utf-8").splitlines() == ["1"]
