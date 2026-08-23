# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os

import pytest

import cuda.coop.cutlass as cutlass_coop
from cuda import coop

from ....support.paths import REPO_ROOT

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("requires a CUDA-capable PyTorch runtime", allow_module_level=True)

from_dlpack = runtime.from_dlpack
Int32 = pytest.importorskip("cutlass.base_dsl.typing").Int32

pytestmark = [
    pytest.mark.backend_cutlass,
    pytest.mark.runtime,
    pytest.mark.gpu,
]

_THREADS = 64
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD
_OUTPUT_SEGMENTS = 9


@pytest.fixture(scope="module", autouse=True)
def _isolated_provider_cache(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp(
        "cuda-coop-cutlass-adjacent-discontinuity-runtime"
    )
    env_values = {
        "CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT": "ltoir",
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR": os.fspath(cache_dir),
        "CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT": os.fspath(REPO_ROOT),
    }
    original = {name: os.environ.get(name) for name in env_values}
    os.environ.update(env_values)
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _store_items(output: cute.Tensor, segment: int, items) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    offset = segment * _TILE_ITEMS + tidx * _ITEMS_PER_THREAD
    output[offset] = items[0]
    output[offset + 1] = items[1]


@cute.kernel
def _adjacent_discontinuity_kernel(values: cute.Tensor, output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    common_items = coop.ThreadData(_ITEMS_PER_THREAD)
    common_items[0] = values[tidx * _ITEMS_PER_THREAD]
    common_items[1] = values[tidx * _ITEMS_PER_THREAD + 1]
    common_block = coop.this_block()
    common_left = coop.adjacent_difference(
        common_block,
        common_items,
        valid_items=Int32(_TILE_ITEMS - 3),
        tile_predecessor_item=Int32(-13),
    )
    common_right = coop.adjacent_difference(
        common_block,
        common_items,
        direction="right",
        tile_successor_item=Int32(29),
    )
    common_heads = coop.discontinuity(
        common_block,
        common_items,
        tile_predecessor_item=Int32(-13),
    )
    common_tails = coop.discontinuity(
        common_block,
        common_items,
        mode="tails",
        tile_successor_item=Int32(29),
    )

    qualified_items = cutlass_coop.ThreadData.from_values(
        common_items[0],
        common_items[1],
        dtype=Int32,
    )
    qualified_block = cutlass_coop.this_block()
    storage = cutlass_coop.TempStorage(
        4096,
        alignment=64,
        auto_sync=False,
    )
    qualified_left = cutlass_coop.adjacent_difference(
        qualified_block,
        qualified_items,
        valid_items=Int32(_TILE_ITEMS - 3),
        tile_predecessor_item=Int32(-13),
        temp_storage=storage,
    )
    storage.sync()
    qualified_right = cutlass_coop.adjacent_difference(
        qualified_block,
        qualified_items,
        direction="right",
        tile_successor_item=Int32(29),
        temp_storage=storage,
    )
    storage.sync()
    qualified_heads, qualified_tails = cutlass_coop.discontinuity(
        qualified_block,
        qualified_items,
        mode="heads_and_tails",
        tile_predecessor_item=Int32(-13),
        tile_successor_item=Int32(29),
        temp_storage=storage,
    )
    storage.sync()

    _store_items(output, 0, common_items)
    _store_items(output, 1, common_left)
    _store_items(output, 2, common_right)
    _store_items(output, 3, common_heads)
    _store_items(output, 4, common_tails)
    _store_items(output, 5, qualified_left)
    _store_items(output, 6, qualified_right)
    _store_items(output, 7, qualified_heads)
    _store_items(output, 8, qualified_tails)


@cute.jit
def _run_adjacent_discontinuity(values: cute.Tensor, output: cute.Tensor):
    _adjacent_discontinuity_kernel(values, output).launch(
        grid=(1, 1, 1),
        block=(_THREADS, 1, 1),
    )


def test_adjacent_discontinuity_matches_both_routes_and_oracle() -> None:
    cutlass.cuda.initialize_cuda_context()
    indices = torch.arange(_TILE_ITEMS, dtype=torch.int32)
    values_host = ((indices * 11 + indices // 3) % 17) - 8
    values = values_host.cuda()
    output = torch.full(
        (_OUTPUT_SEGMENTS * _TILE_ITEMS,),
        -999,
        dtype=torch.int32,
        device="cuda",
    )

    _run_adjacent_discontinuity(from_dlpack(values), from_dlpack(output))
    torch.cuda.synchronize()

    valid_items = _TILE_ITEMS - 3
    expected_left = values_host.clone()
    expected_left[0] = values_host[0] - (-13)
    expected_left[1:valid_items] = (
        values_host[1:valid_items] - values_host[: valid_items - 1]
    )
    expected_right = values_host.clone()
    expected_right[:-1] = values_host[:-1] - values_host[1:]
    expected_right[-1] = values_host[-1] - 29
    expected_heads = torch.empty_like(values_host)
    expected_heads[0] = values_host[0] != -13
    expected_heads[1:] = values_host[1:] != values_host[:-1]
    expected_tails = torch.empty_like(values_host)
    expected_tails[:-1] = values_host[:-1] != values_host[1:]
    expected_tails[-1] = values_host[-1] != 29

    segments = output.cpu().reshape(_OUTPUT_SEGMENTS, _TILE_ITEMS)
    torch.testing.assert_close(segments[0], values_host, atol=0, rtol=0)
    torch.testing.assert_close(segments[1], expected_left, atol=0, rtol=0)
    torch.testing.assert_close(segments[2], expected_right, atol=0, rtol=0)
    torch.testing.assert_close(segments[3], expected_heads, atol=0, rtol=0)
    torch.testing.assert_close(segments[4], expected_tails, atol=0, rtol=0)
    torch.testing.assert_close(segments[5], expected_left, atol=0, rtol=0)
    torch.testing.assert_close(segments[6], expected_right, atol=0, rtol=0)
    torch.testing.assert_close(segments[7], expected_heads, atol=0, rtol=0)
    torch.testing.assert_close(segments[8], expected_tails, atol=0, rtol=0)
    torch.testing.assert_close(values.cpu(), values_host, atol=0, rtol=0)
