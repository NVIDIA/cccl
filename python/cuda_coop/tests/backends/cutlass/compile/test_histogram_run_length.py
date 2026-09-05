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

_BLOCK_THREADS = 32
_BINS = 32
_DECODED_ITEMS_PER_THREAD = 2


@cute.kernel
def _histogram_run_length_kernel(
    values: cute.Tensor,
    lengths: cute.Tensor,
    output: cute.Tensor,
):
    rank = cute.arch.thread_idx()[0]

    common_samples = coop.ThreadData(1, dtype=int)
    qualified_samples = cutlass_coop.ThreadData(1, dtype=Int32)
    common_samples[0] = values[rank]
    qualified_samples[0] = values[rank]
    common_counts = coop.histogram(
        coop.this_block(),
        common_samples,
        bins=_BINS,
    )
    qualified_counts = cutlass_coop.histogram(
        cutlass_coop.this_block(),
        qualified_samples,
        bins=_BINS,
        algorithm="sort",
    )

    common_run_values = coop.ThreadData(1, dtype=int)
    common_run_lengths = coop.ThreadData(1, dtype=int)
    qualified_run_values = cutlass_coop.ThreadData(1, dtype=Int32)
    qualified_run_lengths = cutlass_coop.ThreadData(1, dtype=Int32)
    common_run_values[0] = values[rank]
    common_run_lengths[0] = lengths[rank]
    qualified_run_values[0] = values[rank]
    qualified_run_lengths[0] = lengths[rank]
    common_decoded = coop.run_length_decode(
        coop.this_block(),
        common_run_values,
        common_run_lengths,
        decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
        decoded_window_offset=1,
    )
    relative = cutlass_coop.ThreadData(
        _DECODED_ITEMS_PER_THREAD,
        dtype=Int32,
    )
    total = cutlass_coop.ThreadData(1, dtype=Int32)
    qualified_decoded = cutlass_coop.run_length_decode(
        cutlass_coop.this_block(),
        qualified_run_values,
        qualified_run_lengths,
        decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
        decoded_window_offset=1,
        relative_offsets=relative,
        total_decoded_size=total,
    )

    output[rank] = common_counts[0] + qualified_counts[0]
    base = _BLOCK_THREADS + rank * _DECODED_ITEMS_PER_THREAD
    output[base] = common_decoded[0] + qualified_decoded[0] + relative[0]
    output[base + 1] = common_decoded[1] + qualified_decoded[1] + relative[1]
    output[3 * _BLOCK_THREADS + rank] = total[0]


@cute.jit
def _run_histogram_run_length(
    values: cute.Tensor,
    lengths: cute.Tensor,
    output: cute.Tensor,
):
    _histogram_run_length_kernel(values, lengths, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


def test_common_and_qualified_histogram_run_length_compile_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "provider-cache"
    dump_dir = tmp_path / "provider-dump"
    dump_dir.mkdir()
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT", str(REPO_ROOT))

    fake_values = runtime.make_fake_compact_tensor(Int32, (_BLOCK_THREADS,))
    fake_lengths = runtime.make_fake_compact_tensor(Int32, (_BLOCK_THREADS,))
    fake_output = runtime.make_fake_compact_tensor(Int32, (4 * _BLOCK_THREADS,))
    compiled = cute.compile(
        _run_histogram_run_length,
        fake_values,
        fake_lengths,
        fake_output,
    )

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0

    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    source = sources[0].read_text(encoding="utf-8")
    for header in (
        "cub/block/block_histogram.cuh",
        "cub/block/block_run_length_decode.cuh",
    ):
        assert f"#include <{header}>" in source
    assert source.count("cuda_coop_cutlass_cub_histogram_b32_") == 2
    assert source.count("cuda_coop_cutlass_cub_run_length_decode_b32_") == 2
    assert ".Histogram(items, histogram);" in source
    assert ".DecodeAt(" in source
    assert ".DecodeWithOffsetsAt(" in source
    assert "bool offset_in_range = decoded_offset < decoded_total;" in source
