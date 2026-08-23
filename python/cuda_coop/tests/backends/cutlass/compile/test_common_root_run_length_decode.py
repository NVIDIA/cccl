# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from examples.cutlass._common_root_run_length_decode_codegen_probe import (
    BLOCK_THREADS,
    DECODED_OUTPUT_SEGMENTS,
    TOTAL_OUTPUT_ITEMS,
    TOTAL_RUNS,
    WINDOW_OFFSET_COUNT,
    make_runner,
)

from ....support.paths import REPO_ROOT

cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
Uint64 = pytest.importorskip("cutlass.base_dsl.typing").Uint64

_EXPECTED_SYMBOL = "cuda_coop_cutlass_cub_run_length_decode_b8x4x2_vu64_lu64_r2_x3"
_EXPECTED_OFFSETS_SYMBOL = f"{_EXPECTED_SYMBOL}_offsets"


@pytest.mark.evidence_for(
    "group.run_length_decode",
    backend="cutlass",
    evidence="compile",
)
def test_common_and_qualified_wide_windows_compile_to_one_shared_cub_plan(
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
    fake_values = runtime.make_fake_compact_tensor(Uint64, (TOTAL_RUNS,))
    fake_lengths = runtime.make_fake_compact_tensor(Uint64, (TOTAL_RUNS,))
    fake_offsets = runtime.make_fake_compact_tensor(Uint64, (WINDOW_OFFSET_COUNT,))
    fake_preserved_values = runtime.make_fake_compact_tensor(
        Uint64,
        (2 * TOTAL_RUNS,),
    )
    fake_preserved_lengths = runtime.make_fake_compact_tensor(
        Uint64,
        (2 * TOTAL_RUNS,),
    )
    fake_decoded = runtime.make_fake_compact_tensor(
        Uint64,
        (DECODED_OUTPUT_SEGMENTS * TOTAL_OUTPUT_ITEMS,),
    )
    fake_relative = runtime.make_fake_compact_tensor(
        Uint64,
        (TOTAL_OUTPUT_ITEMS,),
    )
    fake_total = runtime.make_fake_compact_tensor(Uint64, (BLOCK_THREADS,))
    compiled = cute.compile(
        run,
        fake_values,
        fake_lengths,
        fake_offsets,
        fake_preserved_values,
        fake_preserved_lengths,
        fake_decoded,
        fake_relative,
        fake_total,
    )

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0

    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    source = sources[0].read_text(encoding="utf-8")
    assert "#include <cuda/std/type_traits>" in source
    assert source.count(f"{_EXPECTED_SYMBOL}(") == 1
    assert source.count(f"{_EXPECTED_OFFSETS_SYMBOL}(") == 1
    assert "unsigned long long decoded_window_offset" in source
    assert "decoded_window_offset < 0" not in source
    assert "decoded_offset < decoded_total" in source
    assert "decoded_total - decoded_offset : 0ull" in source
    assert "local_target_0 < decoded_remaining" in source
    assert "first_target" not in source
    assert "static_cast<unsigned long long>(~0ull)" in source
