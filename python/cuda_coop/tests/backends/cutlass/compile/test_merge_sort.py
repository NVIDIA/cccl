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

_BLOCK_THREADS = 64
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
_VALID_BLOCK_ITEMS = 117
_VALID_LOGICAL_ITEMS = 13
_OUTPUT_SEGMENTS = 5


def _store(output, segment: int, rank, values) -> None:
    offset = segment * _TILE_ITEMS + rank * _ITEMS_PER_THREAD
    output[offset] = values[0]
    output[offset + 1] = values[1]


@cute.kernel
def _merge_sort_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    output: cute.Tensor,
):
    rank = cute.arch.thread_idx()[0]
    offset = rank * Int32(_ITEMS_PER_THREAD)

    common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=int)
    common_values = coop.ThreadData(_ITEMS_PER_THREAD, dtype=int)
    qualified_keys = cutlass_coop.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
    common_keys[0] = keys_in[offset]
    common_keys[1] = keys_in[offset + Int32(1)]
    common_values[0] = values_in[offset]
    common_values[1] = values_in[offset + Int32(1)]
    qualified_keys[0] = keys_in[offset]
    qualified_keys[1] = keys_in[offset + Int32(1)]

    pair_keys, pair_values = coop.merge_sort_pairs(
        coop.this_block(),
        common_keys,
        common_values,
    )
    fixed_storage = cutlass_coop.TempStorage(4096, alignment=16)
    partial_block = cutlass_coop.merge_sort_keys(
        cutlass_coop.this_block(),
        qualified_keys,
        descending=True,
        valid_items=_VALID_BLOCK_ITEMS,
        oob_default=-2_147_483_648,
        temp_storage=fixed_storage,
    )
    physical_warp = coop.merge_sort_keys(coop.this_warp(), common_keys)
    logical_warp = cutlass_coop.merge_sort_keys(
        cutlass_coop.this_warp().group_by(8),
        qualified_keys,
        valid_items=_VALID_LOGICAL_ITEMS,
        oob_default=2_147_483_647,
    )

    _store(output, 0, rank, pair_keys)
    _store(output, 1, rank, pair_values)
    _store(output, 2, rank, partial_block)
    _store(output, 3, rank, physical_warp)
    _store(output, 4, rank, logical_warp)


@cute.jit
def _run_merge_sort(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    output: cute.Tensor,
):
    _merge_sort_kernel(keys_in, values_in, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


def test_common_and_qualified_merge_sort_compile_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "provider-cache"
    dump_dir = tmp_path / "provider-dump"
    dump_dir.mkdir()
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT", str(REPO_ROOT))

    fake_keys = runtime.make_fake_compact_tensor(Int32, (_TILE_ITEMS,))
    fake_values = runtime.make_fake_compact_tensor(Int32, (_TILE_ITEMS,))
    fake_output = runtime.make_fake_compact_tensor(
        Int32,
        (_OUTPUT_SEGMENTS * _TILE_ITEMS,),
    )
    compiled = cute.compile(
        _run_merge_sort,
        fake_keys,
        fake_values,
        fake_output,
    )

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0

    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    source = sources[0].read_text(encoding="utf-8")
    assert "#include <cub/block/block_merge_sort.cuh>" in source
    assert "#include <cub/warp/warp_merge_sort.cuh>" in source
    for symbol in (
        "cuda_coop_cutlass_cub_merge_sort_block_b64_pairs_ascending_ki32_vi32_x2_full",
        "cuda_coop_cutlass_cub_merge_sort_block_b64_keys_descending_"
        "ki32_x2_partial_external_scratch",
        "cuda_coop_cutlass_cub_merge_sort_warp_b64_w32_keys_ascending_ki32_x2_full",
        "cuda_coop_cutlass_cub_merge_sort_warp_b64_w8_keys_ascending_ki32_x2_partial",
    ):
        assert source.count(f"{symbol}(") == 1
