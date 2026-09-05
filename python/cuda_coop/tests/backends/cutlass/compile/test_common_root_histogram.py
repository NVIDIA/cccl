# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from examples.cutlass._group_histogram_codegen_probe import (
    _COUNTER_CAPACITY,
    _HISTOGRAM_SEGMENTS,
    _SAMPLE_COUNT,
    make_runner,
)

from ....support.paths import REPO_ROOT

cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
types = pytest.importorskip("cutlass.base_dsl.typing")

_EXPECTED_SYMBOLS = {
    "cuda_coop_cutlass_cub_histogram_b8x4x2_atomic_u8_x3_count_i32_bins97_out2",
    "cuda_coop_cutlass_cub_histogram_b8x4x2_sort_u8_x3_count_i64_bins97_out2",
}


@pytest.mark.evidence_for("group.histogram", backend="cutlass", evidence="compile")
def test_common_and_qualified_histogram_compile_to_two_cub_block_plans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "provider-cache"
    dump_dir = tmp_path / "provider-dump"
    dump_dir.mkdir()
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT", str(REPO_ROOT))

    run, *_ = make_runner()
    fake_values = runtime.make_fake_compact_tensor(types.Uint8, (_SAMPLE_COUNT,))
    fake_samples = runtime.make_fake_compact_tensor(
        types.Uint8,
        (2 * _SAMPLE_COUNT,),
    )
    fake_histograms = runtime.make_fake_compact_tensor(
        types.Int64,
        (_HISTOGRAM_SEGMENTS * _COUNTER_CAPACITY,),
    )
    compiled = cute.compile(run, fake_values, fake_samples, fake_histograms)

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0

    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    source = sources[0].read_text(encoding="utf-8")
    for symbol in _EXPECTED_SYMBOLS:
        assert source.count(f"{symbol}(") == 1
