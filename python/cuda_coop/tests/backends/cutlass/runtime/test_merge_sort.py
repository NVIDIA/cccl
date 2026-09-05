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

_BLOCK_THREADS = 64
_WARP_THREADS = 32
_LOGICAL_WARP_THREADS = 8
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
_WARP_ITEMS = _WARP_THREADS * _ITEMS_PER_THREAD
_LOGICAL_ITEMS = _LOGICAL_WARP_THREADS * _ITEMS_PER_THREAD
_VALID_BLOCK_ITEMS = 117
_VALID_LOGICAL_ITEMS = 13
_OUTPUT_SEGMENTS = 6


@pytest.fixture(scope="module", autouse=True)
def _isolated_provider_cache(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("cuda-coop-cutlass-merge-sort-runtime")
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


def _store(output, segment: int, rank, values) -> None:
    offset = segment * _TILE_ITEMS + rank * _ITEMS_PER_THREAD
    output[offset] = values[0]
    output[offset + 1] = values[1]


@cute.kernel
def _merge_sort_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    output: cute.Tensor,
):
    rank = cute.arch.thread_idx()[0]
    offset = rank * Int32(_ITEMS_PER_THREAD)

    common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=int)
    common_values = coop.ThreadData(_ITEMS_PER_THREAD, dtype=int)
    qualified_keys = cutlass_coop.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
    common_keys[0] = keys_in[offset]
    common_keys[1] = keys_in[offset + Int32(1)]
    common_values[0] = values_in[offset]
    common_values[1] = values_in[offset + Int32(1)]
    qualified_keys[0] = keys_in[offset]
    qualified_keys[1] = keys_in[offset + Int32(1)]

    storage = coop.TempStorage(4096, alignment=16)
    pair_keys, pair_values = coop.merge_sort_pairs(
        coop.this_block(),
        common_keys,
        common_values,
        temp_storage=storage,
    )
    partial_block = cutlass_coop.merge_sort_keys(
        cutlass_coop.this_block(),
        qualified_keys,
        descending=True,
        valid_items=_VALID_BLOCK_ITEMS,
        oob_default=-2_147_483_648,
        temp_storage=storage,
    )
    physical_warp = coop.merge_sort_keys(coop.this_warp(), common_keys)
    logical_warp = cutlass_coop.merge_sort_keys(
        cutlass_coop.this_warp().group_by(_LOGICAL_WARP_THREADS),
        qualified_keys,
        valid_items=_VALID_LOGICAL_ITEMS,
        oob_default=2_147_483_647,
    )

    _store(output, 0, rank, common_keys)
    _store(output, 1, rank, pair_keys)
    _store(output, 2, rank, pair_values)
    _store(output, 3, rank, partial_block)
    _store(output, 4, rank, physical_warp)
    _store(output, 5, rank, logical_warp)


@cute.jit
def _run_merge_sort(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    output: cute.Tensor,
):
    _merge_sort_kernel(keys_in, values_in, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


@cute.kernel
def _merge_sort_floating_pairs_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    keys_out: cute.Tensor,
    values_out: cute.Tensor,
):
    rank = cute.arch.thread_idx()[0]
    sorted_key, sorted_value = cutlass_coop.merge_sort_pairs(
        cutlass_coop.this_block(),
        keys_in[rank],
        values_in[rank],
        descending=True,
    )
    keys_out[rank] = sorted_key
    values_out[rank] = sorted_value


@cute.jit
def _run_merge_sort_floating_pairs(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    keys_out: cute.Tensor,
    values_out: cute.Tensor,
):
    _merge_sort_floating_pairs_kernel(
        keys_in,
        values_in,
        keys_out,
        values_out,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1))


def test_block_warp_partial_pairs_and_fixed_storage_match_oracles() -> None:
    cutlass.cuda.initialize_cuda_context()
    indices = torch.arange(_TILE_ITEMS, dtype=torch.int32)
    keys_host = (indices * 37 + 11) % _TILE_ITEMS
    values_host = indices * 10 + 3
    keys = keys_host.cuda()
    values = values_host.cuda()
    output = torch.full(
        (_OUTPUT_SEGMENTS * _TILE_ITEMS,),
        -777_777,
        dtype=torch.int32,
        device="cuda",
    )

    _run_merge_sort(
        from_dlpack(keys),
        from_dlpack(values),
        from_dlpack(output),
    )
    torch.cuda.synchronize()
    observed = output.cpu().reshape(_OUTPUT_SEGMENTS, _TILE_ITEMS)

    torch.testing.assert_close(observed[0], keys_host, atol=0, rtol=0)
    order = torch.argsort(keys_host)
    torch.testing.assert_close(observed[1], keys_host[order], atol=0, rtol=0)
    torch.testing.assert_close(observed[2], values_host[order], atol=0, rtol=0)
    torch.testing.assert_close(
        observed[3, :_VALID_BLOCK_ITEMS],
        torch.sort(
            keys_host[:_VALID_BLOCK_ITEMS],
            descending=True,
        ).values,
        atol=0,
        rtol=0,
    )

    expected_warp = torch.empty_like(keys_host)
    for base in range(0, _TILE_ITEMS, _WARP_ITEMS):
        expected_warp[base : base + _WARP_ITEMS] = torch.sort(
            keys_host[base : base + _WARP_ITEMS]
        ).values
    torch.testing.assert_close(observed[4], expected_warp, atol=0, rtol=0)

    for base in range(0, _TILE_ITEMS, _LOGICAL_ITEMS):
        torch.testing.assert_close(
            observed[5, base : base + _VALID_LOGICAL_ITEMS],
            torch.sort(keys_host[base : base + _VALID_LOGICAL_ITEMS]).values,
            atol=0,
            rtol=0,
        )

    torch.testing.assert_close(keys.cpu(), keys_host, atol=0, rtol=0)
    torch.testing.assert_close(values.cpu(), values_host, atol=0, rtol=0)


@pytest.mark.parametrize("key_dtype", [torch.float32, torch.float64])
def test_qualified_block_sorts_floating_point_pairs(key_dtype) -> None:
    cutlass.cuda.initialize_cuda_context()
    values_host = torch.arange(32, dtype=torch.int32)
    keys_host = ((values_host * 17) % 32).to(key_dtype) + 0.25
    keys = keys_host.cuda()
    values = values_host.cuda()
    keys_out = torch.empty_like(keys)
    values_out = torch.empty_like(values)

    _run_merge_sort_floating_pairs(
        from_dlpack(keys),
        from_dlpack(values),
        from_dlpack(keys_out),
        from_dlpack(values_out),
    )
    torch.cuda.synchronize()

    order = torch.argsort(keys_host, descending=True)
    torch.testing.assert_close(keys_out.cpu(), keys_host[order], atol=0, rtol=0)
    torch.testing.assert_close(values_out.cpu(), values_host[order], atol=0, rtol=0)
