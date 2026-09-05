# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

import cuda.coop.cutlass as cutlass_coop
from cuda import coop

from ....support.paths import REPO_ROOT

cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
Int32 = pytest.importorskip("cutlass.base_dsl.typing").Int32

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]

_THREADS = 64
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD
_OUTPUT_SEGMENTS = 6


@cute.kernel
def _data_movement_kernel(values: cute.Tensor, output: cute.Tensor):
    block = coop.this_block()
    common_items = coop.ThreadData(_ITEMS_PER_THREAD)
    common_loaded = coop.load(block, values, common_items)
    coop.store(block, output, common_loaded)

    qualified_block = cutlass_coop.this_block()
    qualified_items = cutlass_coop.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
    qualified_loaded = cutlass_coop.load(
        qualified_block,
        values,
        qualified_items,
        algorithm="striped",
        offset=_TILE_ITEMS,
    )
    cutlass_coop.store(
        qualified_block,
        output,
        qualified_loaded,
        algorithm="striped",
        offset=_TILE_ITEMS,
    )

    common_exchange = coop.exchange(block, common_loaded)
    coop.store(block, output, common_exchange, offset=2 * _TILE_ITEMS)
    qualified_exchange = cutlass_coop.exchange(
        qualified_block,
        qualified_loaded,
        mode="blocked_to_striped",
    )
    cutlass_coop.store(
        qualified_block,
        output,
        qualified_exchange,
        offset=3 * _TILE_ITEMS,
    )

    common_shuffle = coop.shuffle(block, common_loaded, mode="down")
    coop.store(block, output, common_shuffle, offset=4 * _TILE_ITEMS)
    qualified_shuffle = cutlass_coop.shuffle(
        qualified_block,
        qualified_loaded,
        mode="up",
    )
    cutlass_coop.store(
        qualified_block,
        output,
        qualified_shuffle,
        offset=5 * _TILE_ITEMS,
    )


@cute.jit
def _run_data_movement(values: cute.Tensor, output: cute.Tensor):
    _data_movement_kernel(values, output).launch(
        grid=(1, 1, 1),
        block=(_THREADS, 1, 1),
    )


def test_common_and_qualified_data_movement_compile_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "provider-cache"
    dump_dir = tmp_path / "provider-dump"
    dump_dir.mkdir()
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT", str(REPO_ROOT))

    fake_values = runtime.make_fake_compact_tensor(Int32, (2 * _TILE_ITEMS,))
    fake_output = runtime.make_fake_compact_tensor(
        Int32,
        (_OUTPUT_SEGMENTS * _TILE_ITEMS,),
    )
    compiled = cute.compile(_run_data_movement, fake_values, fake_output)

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0

    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    source = sources[0].read_text(encoding="utf-8")
    for header in (
        "cub/block/block_load.cuh",
        "cub/block/block_store.cuh",
        "cub/block/block_exchange.cuh",
        "cub/block/block_shuffle.cuh",
    ):
        assert f"#include <{header}>" in source
    for symbol_fragment in (
        "cuda_coop_cutlass_cub_load_block_b64_direct_i32_x2_",
        "cuda_coop_cutlass_cub_load_block_b64_striped_i32_x2_",
        "cuda_coop_cutlass_cub_store_block_b64_direct_i32_x2_",
        "cuda_coop_cutlass_cub_store_block_b64_striped_i32_x2_",
        "cuda_coop_cutlass_cub_exchange_block_b64_stripedtoblocked_i32_x2",
        "cuda_coop_cutlass_cub_exchange_block_b64_blockedtostriped_i32_x2",
        "cuda_coop_cutlass_shuffle_b64_down_i32_x2",
        "cuda_coop_cutlass_shuffle_b64_up_i32_x2",
    ):
        assert symbol_fragment in source
