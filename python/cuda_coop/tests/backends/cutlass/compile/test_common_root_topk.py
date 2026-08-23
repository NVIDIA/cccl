# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
from pathlib import Path

import pytest

from examples.cutlass._common_root_topk_codegen_probe import (
    OUTPUT_SEGMENTS,
    TOTAL_ITEMS,
    make_runner,
)

from ....support.paths import REPO_ROOT

cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
Int32 = pytest.importorskip("cutlass.base_dsl.typing").Int32

_EXPECTED_SYMBOLS = {
    "cuda_coop_cutlass_topk_max_keys_i32_bt64_x2",
    "cuda_coop_cutlass_topk_min_keys_i32_bt64_x2",
}


@pytest.mark.evidence_for("group.topk_max_keys", backend="cutlass", evidence="compile")
@pytest.mark.evidence_for("group.topk_min_keys", backend="cutlass", evidence="compile")
def test_common_and_qualified_topk_compile_to_two_shared_cub_plans(
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
    fake_keys = runtime.make_fake_compact_tensor(Int32, (TOTAL_ITEMS,))
    fake_controls = runtime.make_fake_compact_tensor(Int32, (4,))
    fake_output = runtime.make_fake_compact_tensor(
        Int32,
        (OUTPUT_SEGMENTS * TOTAL_ITEMS,),
    )
    compiled = cute.compile(run, fake_keys, fake_controls, fake_output)

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0

    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    source = sources[0].read_text(encoding="utf-8")
    for symbol in _EXPECTED_SYMBOLS:
        assert source.count(f"{symbol}(") == 1
    assert "static_assert(planned_temp_bytes >= required_temp_bytes" in source
    assert "alignof(TempStorageT)" in source


@pytest.mark.evidence_for("group.topk_max_keys", backend="cutlass", evidence="compile")
@pytest.mark.evidence_for("group.topk_min_keys", backend="cutlass", evidence="compile")
def test_topk_provider_and_scratch_proof_compile_with_nvrtc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("cuda.bindings.nvrtc")
    cutlass_typing = pytest.importorskip("cutlass.base_dsl.typing")

    from cuda.coop.cutlass._dsl import _provider_bundle as provider_bundle
    from cuda.coop.cutlass._dsl.block import _provider as block_provider

    requests = [
        block_provider._ShimRequest(
            kind="topk_keys",
            op="max",
            key_type=key_type,
            items_per_thread=2,
            block_threads=64,
        )
        for key_type in (
            cutlass_typing.Uint8,
            cutlass_typing.Int32,
            cutlass_typing.Uint32,
            cutlass_typing.Int64,
            cutlass_typing.Uint64,
            cutlass_typing.Float32,
            cutlass_typing.Float64,
        )
    ]
    requests.append(
        block_provider._ShimRequest(
            kind="topk_keys",
            op="min",
            key_type=Int32,
            items_per_thread=2,
            block_threads=64,
        )
    )
    source = block_provider._render_bundle_source(requests)
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT", "ltoir")
    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR",
        str(tmp_path / "provider-cache"),
    )
    bundle_path = provider_bundle.compile_bundle_source(
        source,
        scope=block_provider._SCOPE,
        provider_dir=os.path.dirname(block_provider.__file__),
        registered_headers=block_provider._registered_cccl_headers,
        select_bundle_format=lambda: "ltoir",
        resolve_nvrtc_sm_arch=lambda: "sm_120",
        resolve_nvrtc_arch=lambda: "compute_120",
    )

    assert source.count(
        "static_assert(planned_temp_bytes >= required_temp_bytes"
    ) == len(requests)
    assert bundle_path.endswith(".ltoir")
    assert os.path.getsize(bundle_path) > 0
