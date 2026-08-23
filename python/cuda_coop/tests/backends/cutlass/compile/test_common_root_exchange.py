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

_THREADS = 64
_ITEMS_PER_THREAD = 5
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD


def _store_items(output: cute.Tensor, segment: int, items) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    offset = segment * _TILE_ITEMS + tidx * _ITEMS_PER_THREAD
    output[offset] = items[0]
    output[offset + 1] = items[1]
    output[offset + 2] = items[2]
    output[offset + 3] = items[3]
    output[offset + 4] = items[4]


@cute.kernel
def _common_exchange_kernel(values: cute.Tensor, output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    offset = tidx * _ITEMS_PER_THREAD
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
    items[0] = values[offset]
    items[1] = values[offset + 1]
    items[2] = values[offset + 2]
    items[3] = values[offset + 3]
    items[4] = values[offset + 4]

    block = coop.this_block()
    _store_items(output, 0, coop.exchange(block, items))
    _store_items(
        output,
        1,
        coop.exchange(block, items, mode="blocked_to_striped"),
    )
    warp = coop.this_warp()
    _store_items(output, 2, coop.exchange(warp, items))
    _store_items(
        output,
        3,
        coop.exchange(warp, items, mode="blocked_to_striped"),
    )


@cute.jit
def _run_common_exchange(values: cute.Tensor, output: cute.Tensor):
    _common_exchange_kernel(values, output).launch(
        grid=(1, 1, 1),
        block=(_THREADS, 1, 1),
    )


@pytest.mark.evidence_for("group.exchange", backend="cutlass", evidence="compile")
def test_common_exchange_compiles_for_block_and_two_physical_warps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "provider-cache"
    dump_dir = tmp_path / "provider-dump"
    dump_dir.mkdir()
    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR",
        str(cache_dir),
    )
    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR",
        str(dump_dir),
    )
    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT",
        str(REPO_ROOT),
    )

    fake_values = runtime.make_fake_compact_tensor(Int32, (_TILE_ITEMS,))
    fake_output = runtime.make_fake_compact_tensor(Int32, (4 * _TILE_ITEMS,))
    compiled = cute.compile(_run_common_exchange, fake_values, fake_output)

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0

    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    source = sources[0].read_text(encoding="utf-8")
    expected_symbols = {
        "cuda_coop_cutlass_cub_exchange_block_b64_stripedtoblocked_i32_x5",
        "cuda_coop_cutlass_cub_exchange_block_b64_blockedtostriped_i32_x5",
        "cuda_coop_cutlass_cub_exchange_warp_b64_stripedtoblocked_i32_x5",
        "cuda_coop_cutlass_cub_exchange_warp_b64_blockedtostriped_i32_x5",
    }
    for symbol in expected_symbols:
        assert source.count(f"{symbol}(") == 1
