# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from cuda import coop

from ....support.paths import REPO_ROOT

pytestmark = pytest.mark.usefixtures("qualified_cutlass_backend")

cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
Int32 = pytest.importorskip("cutlass.base_dsl.typing").Int32


@cute.kernel
def _common_load_store_kernel(
    values_in: cute.Tensor,
    block_out: cute.Tensor,
    warp_out: cute.Tensor,
):
    block = coop.this_block()
    block_items = coop.ThreadData(2)
    loaded_block = coop.load(block, values_in, block_items)
    coop.store(block, block_out, loaded_block)

    warp = coop.this_warp()
    warp_items = coop.ThreadData(2)
    loaded_warp = coop.load(warp, values_in, warp_items, algorithm="striped")
    coop.store(warp, warp_out, loaded_warp, algorithm="striped")


@cute.jit
def _run_common_load_store(
    values_in: cute.Tensor,
    block_out: cute.Tensor,
    warp_out: cute.Tensor,
):
    _common_load_store_kernel(values_in, block_out, warp_out).launch(
        grid=(1, 1, 1),
        block=(64, 1, 1),
    )


@pytest.mark.evidence_for("group.load", backend="cutlass", evidence="compile")
@pytest.mark.evidence_for("group.store", backend="cutlass", evidence="compile")
def test_common_load_store_compiles_for_block_and_physical_warp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR",
        str(tmp_path / "provider-cache"),
    )
    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT",
        str(REPO_ROOT),
    )

    fake_values = runtime.make_fake_compact_tensor(Int32, (128,))
    compiled = tuple(
        cute.compile(
            _run_common_load_store,
            fake_values,
            fake_values,
            fake_values,
        )
        for _ in range(2)
    )

    assert all(callable(result) for result in compiled)


def test_portable_root_sum_example_compiles_from_its_source_module(
    source_examples: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del source_examples
    from examples.cutlass import portable_root_sum

    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR",
        str(tmp_path / "provider-cache"),
    )
    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT",
        str(REPO_ROOT),
    )

    run, *_ = portable_root_sum.make_runner()
    fake_values = runtime.make_fake_compact_tensor(
        Int32,
        (portable_root_sum.TILE_ITEMS,),
    )
    fake_totals = runtime.make_fake_compact_tensor(
        Int32,
        (1,),
    )
    compiled = cute.compile(run, fake_values, fake_values, fake_totals)

    assert callable(compiled)
