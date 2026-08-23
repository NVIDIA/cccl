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
_OUTPUT_SEGMENTS = 6


def _store(output, segment: int, rank, values) -> None:
    offset = segment * _TILE_ITEMS + rank * _ITEMS_PER_THREAD
    output[offset] = values[0]
    output[offset + 1] = values[1]


@cute.kernel
def _radix_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    output: cute.Tensor,
):
    rank = cute.arch.thread_idx()[0]
    offset = rank * Int32(_ITEMS_PER_THREAD)

    common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=int)
    qualified_keys = cutlass_coop.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
    qualified_values = cutlass_coop.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
    common_keys[0] = keys_in[offset]
    common_keys[1] = keys_in[offset + Int32(1)]
    qualified_keys[0] = keys_in[offset]
    qualified_keys[1] = keys_in[offset + Int32(1)]
    qualified_values[0] = values_in[offset]
    qualified_values[1] = values_in[offset + Int32(1)]

    common_sorted = coop.radix_sort_keys(
        coop.this_block(),
        common_keys,
        begin_bit=Int32(4),
        end_bit=Int32(12),
    )
    fixed_storage = cutlass_coop.TempStorage(8192, alignment=16)
    pair_keys, pair_values = cutlass_coop.radix_sort_pairs(
        cutlass_coop.this_block(),
        qualified_keys,
        qualified_values,
        begin_bit=Int32(4),
        end_bit=Int32(12),
        descending=True,
        temp_storage=fixed_storage,
    )
    prefix = cutlass_coop.ThreadData(1, dtype=Int32)
    ranks = cutlass_coop.radix_rank(
        cutlass_coop.this_block(),
        qualified_keys,
        begin_bit=24,
        radix_bits=4,
        exclusive_digit_prefix=prefix,
    )

    _store(output, 0, rank, common_keys)
    _store(output, 1, rank, common_sorted)
    _store(output, 2, rank, pair_keys)
    _store(output, 3, rank, pair_values)
    _store(output, 4, rank, ranks)
    prefix_offset = 5 * _TILE_ITEMS + rank * _ITEMS_PER_THREAD
    output[prefix_offset] = prefix[0]
    output[prefix_offset + Int32(1)] = prefix[0]


@cute.jit
def _run_radix(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    output: cute.Tensor,
):
    _radix_kernel(keys_in, values_in, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


def test_common_and_qualified_radix_compile_together(
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
    compiled = cute.compile(_run_radix, fake_keys, fake_values, fake_output)

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0

    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    source = sources[0].read_text(encoding="utf-8")
    assert "#include <cub/block/block_radix_sort.cuh>" in source
    assert "#include <cub/block/block_radix_rank.cuh>" in source
    assert source.count("cuda_coop_cutlass_radix_sort_keys_b64_i32_asc_x2(") == 1
    assert (
        source.count(
            "cuda_coop_cutlass_radix_sort_pairs_i32_b64_i32_desc_x2_external_scratch("
        )
        == 1
    )
    assert source.count("cuda_coop_cutlass_radix_rank_b64_i32_asc_b24_28_x2") == 1
    assert "unsigned int temp_storage_smem_addr" in source
    assert "temp_storage_bytes < required_temp_bytes" in source
    assert "required_temp_alignment - 1ull" in source
    assert "if (temp_storage_auto_sync != 0)" in source
