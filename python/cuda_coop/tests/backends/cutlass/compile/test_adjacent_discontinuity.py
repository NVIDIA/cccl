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
_OUTPUT_SEGMENTS = 8


def _store_items(output: cute.Tensor, segment: int, items) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    offset = segment * _TILE_ITEMS + tidx * _ITEMS_PER_THREAD
    output[offset] = items[0]
    output[offset + 1] = items[1]


@cute.kernel
def _adjacent_discontinuity_kernel(values: cute.Tensor, output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    common_items = coop.ThreadData(_ITEMS_PER_THREAD)
    common_items[0] = values[tidx * _ITEMS_PER_THREAD]
    common_items[1] = values[tidx * _ITEMS_PER_THREAD + 1]
    common_block = coop.this_block()
    common_left = coop.adjacent_difference(
        common_block,
        common_items,
        valid_items=Int32(_TILE_ITEMS - 3),
        tile_predecessor_item=Int32(-13),
    )
    common_heads = coop.discontinuity(
        common_block,
        common_items,
        tile_predecessor_item=Int32(-13),
    )

    qualified_items = cutlass_coop.ThreadData.from_values(
        common_items[0],
        common_items[1],
        dtype=Int32,
    )
    qualified_block = cutlass_coop.this_block()
    storage = cutlass_coop.TempStorage(
        4096,
        alignment=64,
        auto_sync=False,
    )
    qualified_left = cutlass_coop.adjacent_difference(
        qualified_block,
        qualified_items,
        valid_items=Int32(_TILE_ITEMS - 3),
        tile_predecessor_item=Int32(-13),
        temp_storage=storage,
    )
    storage.sync()
    qualified_right = cutlass_coop.adjacent_difference(
        qualified_block,
        qualified_items,
        direction="right",
        tile_successor_item=Int32(29),
        temp_storage=storage,
    )
    storage.sync()
    qualified_heads, qualified_tails = cutlass_coop.discontinuity(
        qualified_block,
        qualified_items,
        mode="heads_and_tails",
        tile_predecessor_item=Int32(-13),
        tile_successor_item=Int32(29),
        temp_storage=storage,
    )
    storage.sync()

    _store_items(output, 0, common_items)
    _store_items(output, 1, common_left)
    _store_items(output, 2, common_heads)
    _store_items(output, 3, qualified_items)
    _store_items(output, 4, qualified_left)
    _store_items(output, 5, qualified_right)
    _store_items(output, 6, qualified_heads)
    _store_items(output, 7, qualified_tails)


@cute.jit
def _run_adjacent_discontinuity(values: cute.Tensor, output: cute.Tensor):
    _adjacent_discontinuity_kernel(values, output).launch(
        grid=(1, 1, 1),
        block=(_THREADS, 1, 1),
    )


def test_common_and_qualified_adjacent_discontinuity_compile_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "provider-cache"
    dump_dir = tmp_path / "provider-dump"
    dump_dir.mkdir()
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT", str(REPO_ROOT))

    fake_values = runtime.make_fake_compact_tensor(Int32, (_TILE_ITEMS,))
    fake_output = runtime.make_fake_compact_tensor(
        Int32,
        (_OUTPUT_SEGMENTS * _TILE_ITEMS,),
    )
    compiled = cute.compile(_run_adjacent_discontinuity, fake_values, fake_output)

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0

    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    source = sources[0].read_text(encoding="utf-8")
    for header in (
        "cub/block/block_adjacent_difference.cuh",
        "cub/block/block_discontinuity.cuh",
    ):
        assert f"#include <{header}>" in source
    for symbol_fragment in (
        "adjacent_difference_b64_subtract_left_i32_x2_partial_predecessor",
        "adjacent_difference_b64_subtract_right_i32_x2_successor_external_scratch",
        "discontinuity_b64_heads_i32_x2_predecessor",
        "discontinuity_b64_heads_and_tails_i32_x2_predecessor_successor_"
        "external_scratch",
    ):
        assert symbol_fragment in source
    assert "temp_storage_bytes < required_temp_bytes" in source
    assert "temp_storage_smem_addr &" in source
    assert "if (temp_storage_auto_sync != 0)" in source
