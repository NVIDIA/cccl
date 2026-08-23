# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from examples.cutlass._common_root_radix_rank_codegen_probe import (
    TOTAL_ITEMS,
    make_dtype_runner,
)

from ....support.paths import REPO_ROOT

cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
Int32 = pytest.importorskip("cutlass.base_dsl.typing").Int32

_EXPECTED_SYMBOLS = {
    "cuda_coop_cutlass_radix_rank_b8x4x2_i32_asc_b24_32_x2",
    "cuda_coop_cutlass_radix_rank_b8x4x2_i32_desc_b24_32_x2",
}


@pytest.mark.evidence_for("group.radix_rank", backend="cutlass", evidence="compile")
def test_common_and_qualified_radix_rank_compile_to_two_shared_cub_plans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "provider-cache"
    dump_dir = tmp_path / "provider-dump"
    dump_dir.mkdir()
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT", str(REPO_ROOT))

    run, *_ = make_dtype_runner("int32")
    fake_keys = runtime.make_fake_compact_tensor(Int32, (TOTAL_ITEMS,))
    fake_preserved = runtime.make_fake_compact_tensor(
        Int32,
        (2 * TOTAL_ITEMS,),
    )
    fake_ranks = runtime.make_fake_compact_tensor(
        Int32,
        (4 * TOTAL_ITEMS,),
    )
    compiled = cute.compile(run, fake_keys, fake_preserved, fake_ranks)

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0

    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    source = sources[0].read_text(encoding="utf-8")
    for symbol in _EXPECTED_SYMBOLS:
        assert source.count(f"{symbol}(") == 1
    assert "^ 0x80000000u" in source
