# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from examples.cutlass._common_root_pairs_codegen_probe import (
    RESULT_SEGMENTS,
    TOTAL_ITEMS,
    make_runner,
)

from ....support.paths import REPO_ROOT

cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
cutlass_typing = pytest.importorskip("cutlass.base_dsl.typing")


@pytest.mark.evidence_for(
    "group.merge_sort_pairs", backend="cutlass", evidence="compile"
)
@pytest.mark.evidence_for(
    "group.radix_sort_pairs", backend="cutlass", evidence="compile"
)
@pytest.mark.evidence_for("group.topk_max_pairs", backend="cutlass", evidence="compile")
@pytest.mark.evidence_for("group.topk_min_pairs", backend="cutlass", evidence="compile")
def test_common_and_qualified_pairs_compile_to_shared_cub_plans(
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
    fake_keys = runtime.make_fake_compact_tensor(cutlass_typing.Int32, (TOTAL_ITEMS,))
    fake_values = runtime.make_fake_compact_tensor(cutlass_typing.Int64, (TOTAL_ITEMS,))
    fake_key_output = runtime.make_fake_compact_tensor(
        cutlass_typing.Int32, (RESULT_SEGMENTS * TOTAL_ITEMS,)
    )
    fake_value_output = runtime.make_fake_compact_tensor(
        cutlass_typing.Int64, (RESULT_SEGMENTS * TOTAL_ITEMS,)
    )
    compiled = cute.compile(
        run,
        fake_keys,
        fake_values,
        fake_key_output,
        fake_value_output,
    )

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0
    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    source = sources[0].read_text(encoding="utf-8")
    assert source.count("merge_sort") >= 4
    assert source.count("radix_sort_pairs") >= 2
    assert source.count("topk_") >= 2
