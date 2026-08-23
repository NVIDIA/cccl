# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from cuda import coop

from ....support.paths import REPO_ROOT

cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
Int32 = pytest.importorskip("cutlass.base_dsl.typing").Int32

_BLOCK_THREADS = 64


def _store_items(output: cute.Tensor, offset: int, items) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    output[offset + tidx * 2] = items[0]
    output[offset + tidx * 2 + 1] = items[1]


@cute.kernel
def _common_sum_scan_kernel(values: cute.Tensor, output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    block = coop.this_block()
    storage = coop.TempStorage()
    items = coop.ThreadData(2, dtype=Int32)
    items[0] = values[tidx * 2]
    items[1] = values[tidx * 2 + 1]

    _store_items(output, 0, coop.scan(block, items, temp_storage=storage))
    _store_items(
        output,
        128,
        coop.scan(block, items, mode="inclusive", temp_storage=storage),
    )
    _store_items(output, 256, coop.exclusive_sum(block, items, temp_storage=storage))
    _store_items(output, 384, coop.inclusive_sum(block, items, temp_storage=storage))
    _store_items(
        output,
        512,
        coop.exclusive_scan(
            block,
            items,
            scan_op="max",
            initial_value=-2_147_483_648,
            temp_storage=storage,
        ),
    )
    _store_items(
        output,
        640,
        coop.inclusive_scan(block, items, scan_op="max", temp_storage=storage),
    )

    value = values[tidx]
    output[768 + tidx] = coop.sum(block, items)
    partial_block_sum = coop.sum(
        block,
        value,
        broadcast=False,
        valid_items=47,
        algorithm="raking",
    )
    if tidx == 0:
        output[832] = partial_block_sum
    block_max = coop.reduce(
        block,
        value,
        binary_op="max",
        broadcast=False,
        algorithm="raking",
    )
    if tidx == 0:
        output[1536] = block_max

    warp = coop.this_warp()
    output[896 + tidx] = coop.scan(warp, value)
    output[960 + tidx] = coop.scan(warp, value, mode="inclusive")
    output[1024 + tidx] = coop.exclusive_sum(warp, value)
    output[1088 + tidx] = coop.inclusive_sum(warp, value)
    output[1152 + tidx] = coop.exclusive_scan(
        warp,
        value,
        scan_op="max",
        initial_value=-2_147_483_648,
    )
    output[1216 + tidx] = coop.inclusive_scan(warp, value, scan_op="max")
    output[1280 + tidx] = coop.sum(warp, value)

    partial_warp_sum = coop.sum(
        warp,
        value,
        broadcast=False,
        valid_items=24,
    )
    lane = tidx % 32
    if lane == 0:
        output[1344 + tidx // 32] = partial_warp_sum
    warp_max = coop.reduce(
        warp,
        value,
        binary_op="max",
        broadcast=False,
        valid_items=24,
    )
    if lane == 0:
        output[1600 + tidx // 32] = warp_max

    output[1408 + tidx] = coop.sum(warp.group_by(8), value)
    output[1472 + tidx] = coop.sum(coop.this_thread(), value)


@cute.jit
def _run_common_sum_scan(values: cute.Tensor, output: cute.Tensor):
    _common_sum_scan_kernel(values, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


@pytest.mark.evidence_for("group.reduce", backend="cutlass", evidence="compile")
@pytest.mark.evidence_for("group.sum", backend="cutlass", evidence="compile")
@pytest.mark.evidence_for("group.scan", backend="cutlass", evidence="compile")
@pytest.mark.evidence_for("group.exclusive_sum", backend="cutlass", evidence="compile")
@pytest.mark.evidence_for("group.inclusive_sum", backend="cutlass", evidence="compile")
@pytest.mark.evidence_for("group.exclusive_scan", backend="cutlass", evidence="compile")
@pytest.mark.evidence_for("group.inclusive_scan", backend="cutlass", evidence="compile")
def test_common_reduce_sum_scan_compiles_for_block_warp_and_mapped_groups(
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

    fake_values = runtime.make_fake_compact_tensor(Int32, (1_664,))
    compiled = cute.compile(_run_common_sum_scan, fake_values, fake_values)

    assert callable(compiled)
