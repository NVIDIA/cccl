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
typing = pytest.importorskip("cutlass.base_dsl.typing")

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]

_THREADS = 64
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD


@cute.kernel
def _topk_kernel(source: cute.Tensor, output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    keys = coop.ThreadData(_ITEMS_PER_THREAD)
    values = cutlass_coop.ThreadData(_ITEMS_PER_THREAD, dtype=typing.Float32)
    keys[0] = source[tidx * _ITEMS_PER_THREAD]
    keys[1] = source[tidx * _ITEMS_PER_THREAD + 1]
    values[0] = typing.Float32(source[tidx * _ITEMS_PER_THREAD])
    values[1] = typing.Float32(source[tidx * _ITEMS_PER_THREAD + 1])

    largest = coop.topk_max_keys(
        coop.this_block(),
        keys,
        4,
        valid_items=_TILE_ITEMS - 1,
        begin_bit=1,
        end_bit=31,
    )
    storage = cutlass_coop.TempStorage(16_384, alignment=16)
    pair_keys, _pair_values = cutlass_coop.topk_min_pairs(
        cutlass_coop.this_block(),
        keys,
        values,
        4,
        begin_bit=1,
        temp_storage=storage,
    )
    output[tidx * _ITEMS_PER_THREAD] = largest[0]
    output[tidx * _ITEMS_PER_THREAD + 1] = pair_keys[0]


@cute.jit
def _run_topk(source: cute.Tensor, output: cute.Tensor):
    _topk_kernel(source, output).launch(
        grid=(1, 1, 1),
        block=(_THREADS, 1, 1),
    )


def test_common_and_qualified_topk_compile_in_one_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "provider-cache"
    dump_dir = tmp_path / "provider-dump"
    dump_dir.mkdir()
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT", str(REPO_ROOT))

    source = runtime.make_fake_compact_tensor(typing.Int32, (_TILE_ITEMS,))
    output = runtime.make_fake_compact_tensor(typing.Int32, (_TILE_ITEMS,))
    compiled = cute.compile(_run_topk, source, output)

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0
    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    provider_source = sources[0].read_text(encoding="utf-8")
    assert provider_source.count("#include <cub/block/block_topk.cuh>") == 1
    assert "cuda_coop_cutlass_cub_topk_max_keys_i32_b64_x2_internal" in (
        provider_source
    )
    assert "cuda_coop_cutlass_cub_topk_min_pairs_f32_i32_b64_x2_external" in (
        provider_source
    )
