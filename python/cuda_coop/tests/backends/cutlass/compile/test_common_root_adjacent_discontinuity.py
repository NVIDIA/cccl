# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

import cuda.coop.cutlass as cutlass_coop
from cuda import coop
from examples.cutlass._group_adjacent_discontinuity_codegen_probe import (
    _SEGMENT_COUNT,
    _TILE_ITEMS,
    make_runner,
)

from ....support.paths import REPO_ROOT

cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
Int32 = pytest.importorskip("cutlass.base_dsl.typing").Int32

_EXPECTED_SYMBOLS = {
    "cuda_coop_cutlass_adjacent_difference_b8x4x2_"
    "subtract_left_i32_x2_partial_predecessor_external_scratch",
    "cuda_coop_cutlass_adjacent_difference_b8x4x2_"
    "subtract_right_i32_x2_successor_external_scratch",
    "cuda_coop_cutlass_discontinuity_b8x4x2_heads_i32_x2_predecessor_external_scratch",
    "cuda_coop_cutlass_discontinuity_b8x4x2_tails_i32_x2_successor_external_scratch",
}


@cute.kernel
def _common_scalar_adjacent_difference_kernel(output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    value = Int32(tidx)
    output[tidx] = coop.adjacent_difference(coop.this_block(), value)


@cute.kernel
def _common_scalar_discontinuity_kernel(output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    value = Int32(tidx)
    output[tidx] = coop.discontinuity(coop.this_block(), value)


@cute.kernel
def _qualified_scalar_comparison_kernel(output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    value = Int32(tidx)
    group = cutlass_coop.this_block()
    difference = cutlass_coop.adjacent_difference(group, value)
    flag = cutlass_coop.discontinuity(group, value)
    output[tidx] = difference + flag


@cute.jit
def _run_common_scalar_adjacent_difference(output: cute.Tensor):
    _common_scalar_adjacent_difference_kernel(output).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


@cute.jit
def _run_common_scalar_discontinuity(output: cute.Tensor):
    _common_scalar_discontinuity_kernel(output).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


@cute.jit
def _run_qualified_scalar_comparison(output: cute.Tensor):
    _qualified_scalar_comparison_kernel(output).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


@pytest.mark.evidence_for(
    "group.adjacent_difference", backend="cutlass", evidence="compile"
)
@pytest.mark.evidence_for("group.discontinuity", backend="cutlass", evidence="compile")
def test_common_and_qualified_comparison_cohort_compiles_to_four_cub_plans(
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
    fake_values = runtime.make_fake_compact_tensor(Int32, (_TILE_ITEMS,))
    fake_output = runtime.make_fake_compact_tensor(
        Int32,
        (_SEGMENT_COUNT * _TILE_ITEMS,),
    )
    compiled = cute.compile(run, fake_values, fake_output)

    assert callable(compiled)
    artifacts = tuple(cache_dir.glob("*.ltoir"))
    assert len(artifacts) == 1
    assert artifacts[0].stat().st_size > 0

    sources = tuple(dump_dir.glob("cuda_coop_cutlass_bundle_*.cpp"))
    assert len(sources) == 1
    source = sources[0].read_text(encoding="utf-8")
    for symbol in _EXPECTED_SYMBOLS:
        assert source.count(f"{symbol}(") == 1


@pytest.mark.parametrize(
    ("rejected", "operation_name"),
    [
        (_run_common_scalar_adjacent_difference, "adjacent_difference"),
        (_run_common_scalar_discontinuity, "discontinuity"),
    ],
)
def test_common_scalar_comparison_rejection_leaves_qualified_compile_usable(
    rejected,
    operation_name: str,
) -> None:
    fake_output = runtime.make_fake_compact_tensor(Int32, (32,))

    with pytest.raises(
        TypeError,
        match=rf"cuda\.coop\.{operation_name} requires a fixed-size ThreadData",
    ):
        cute.compile(rejected, fake_output)

    assert cute.compile(_run_qualified_scalar_comparison, fake_output) is not None
