# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from cuda import coop

from ....support.paths import REPO_ROOT

cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
Int32 = pytest.importorskip("cutlass.base_dsl.typing").Int32
cutlass_coop = pytest.importorskip("cuda.coop.cutlass")

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]

_THREADS = 64
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD


@cute.kernel
def _reduce_scan_kernel(values: cute.Tensor, output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    block = coop.this_block()
    items = coop.ThreadData(_ITEMS_PER_THREAD)
    items[0] = values[tidx * _ITEMS_PER_THREAD]
    items[1] = values[tidx * _ITEMS_PER_THREAD + 1]

    full_sum = coop.sum(block, items)
    inclusive = coop.inclusive_sum(block, items)
    output[tidx * _ITEMS_PER_THREAD] = inclusive[0]
    output[tidx * _ITEMS_PER_THREAD + 1] = inclusive[1]

    qualified_block = cutlass_coop.this_block()
    qualified_items = cutlass_coop.ThreadData.from_values(
        items[0],
        items[1],
        dtype=Int32,
    )
    fixed_storage = cutlass_coop.TempStorage(4096, alignment=16)
    aggregate = cutlass_coop.ThreadData(1, dtype=Int32)
    exclusive = cutlass_coop.exclusive_sum(
        qualified_block,
        qualified_items,
        temp_storage=fixed_storage,
        aggregate_output=aggregate,
    )
    output[_TILE_ITEMS + tidx * _ITEMS_PER_THREAD] = exclusive[0]
    output[_TILE_ITEMS + tidx * _ITEMS_PER_THREAD + 1] = exclusive[1]

    direct_items_sum = cutlass_coop.sum(
        qualified_block,
        qualified_items,
        broadcast=False,
        algorithm="raking",
    )
    mapped_sum = cutlass_coop.sum(
        qualified_block.group_by(1),
        values[tidx],
    )
    logical = coop.this_warp().group_by(8)
    partial_sum = coop.sum(
        logical,
        values[tidx],
        broadcast=False,
        valid_items=7,
    )
    if tidx == 0:
        output[2 * _TILE_ITEMS] = full_sum
        output[2 * _TILE_ITEMS + 1] = direct_items_sum
        output[2 * _TILE_ITEMS + 2] = aggregate[0]
    output[2 * _TILE_ITEMS + 3 + tidx] = mapped_sum
    if tidx % 8 == 0:
        output[3 * _TILE_ITEMS + tidx] = partial_sum


@cute.jit
def _run_reduce_scan(values: cute.Tensor, output: cute.Tensor):
    _reduce_scan_kernel(values, output).launch(
        grid=(1, 1, 1),
        block=(_THREADS, 1, 1),
    )


def test_common_and_qualified_reduce_scan_compile_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "provider-cache"
    dump_dir = tmp_path / "provider-dump"
    dump_dir.mkdir()
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT", "ltoir")
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT", str(REPO_ROOT))

    fake_values = runtime.make_fake_compact_tensor(Int32, (_TILE_ITEMS,))
    fake_output = runtime.make_fake_compact_tensor(Int32, (4 * _TILE_ITEMS,))
    compiled = cute.compile(_run_reduce_scan, fake_values, fake_output)

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0

    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    source = sources[0].read_text(encoding="utf-8")
    assert "#define _CUDAX_DISABLE_COOPERATIVE_GROUPS_INTEROP" in source
    for header in (
        "cuda/experimental/coop.cuh",
        "cub/block/block_reduce.cuh",
        "cub/block/block_scan.cuh",
        "cub/warp/warp_reduce.cuh",
    ):
        assert f"#include <{header}>" in source
    for symbol_fragment in (
        "cuda_coop_cutlass_cudax_reduce_block_b64_sum_i32_x2",
        "cuda_coop_cutlass_cudax_reduce_warps_within_block_1_all_b64_sum_i32",
        "cuda_coop_cutlass_cub_reduce_block_b64_sum_i32_x2_raking_full",
        "cuda_coop_cutlass_cub_reduce_threads_within_warp_8_all_b64_sum_i32",
        "cuda_coop_cutlass_cub_scan_block_b64_exclusivesum_sum_i32_x2",
        "cuda_coop_cutlass_cub_scan_block_b64_inclusivesum_sum_i32_x2",
    ):
        assert symbol_fragment in source
    assert "temp_storage_bytes < required_temp_bytes" in source
