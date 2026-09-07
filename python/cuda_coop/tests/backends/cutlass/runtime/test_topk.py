# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os

import numpy as np
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

Int32 = pytest.importorskip("cutlass.base_dsl.typing").Int32

pytestmark = [
    pytest.mark.backend_cutlass,
    pytest.mark.runtime,
    pytest.mark.gpu,
]

_THREADS = 32
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD


@pytest.fixture(scope="module", autouse=True)
def _isolated_provider_cache(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("cuda-coop-cutlass-topk-runtime")
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


@cute.kernel
def _topk_kernel(
    keys_source: cute.Tensor, values_source: cute.Tensor, output: cute.Tensor
):
    tidx, _, _ = cute.arch.thread_idx()
    offset = tidx * _ITEMS_PER_THREAD
    keys = coop.ThreadData(_ITEMS_PER_THREAD)
    values = cutlass_coop.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
    keys[0] = keys_source[offset]
    keys[1] = keys_source[offset + 1]
    values[0] = values_source[offset]
    values[1] = values_source[offset + 1]

    selected_max = coop.topk_max_keys(
        coop.this_block(),
        keys,
        11,
        valid_items=_TILE_ITEMS - 7,
        begin_bit=1,
        end_bit=32,
    )
    storage = cutlass_coop.TempStorage(16_384, alignment=16)
    selected_min_keys, selected_min_values = cutlass_coop.topk_min_pairs(
        cutlass_coop.this_block(),
        keys,
        values,
        11,
        valid_items=_TILE_ITEMS - 7,
        begin_bit=1,
        temp_storage=storage,
    )
    output[offset] = selected_max[0]
    output[offset + 1] = selected_max[1]
    output[_TILE_ITEMS + offset] = selected_min_keys[0]
    output[_TILE_ITEMS + offset + 1] = selected_min_keys[1]
    output[2 * _TILE_ITEMS + offset] = selected_min_values[0]
    output[2 * _TILE_ITEMS + offset + 1] = selected_min_values[1]
    output[3 * _TILE_ITEMS + offset] = keys[0]
    output[3 * _TILE_ITEMS + offset + 1] = keys[1]
    output[4 * _TILE_ITEMS + offset] = values[0]
    output[4 * _TILE_ITEMS + offset + 1] = values[1]


@cute.jit
def _run_topk(keys: cute.Tensor, values: cute.Tensor, output: cute.Tensor):
    _topk_kernel(keys, values, output).launch(
        grid=(1, 1, 1),
        block=(_THREADS, 1, 1),
    )


def _ordered_digits(values):
    unsigned = values.view(np.uint32).copy()
    unsigned ^= np.uint32(1 << 31)
    return unsigned >> np.uint32(1)


def test_topk_keys_and_pairs_match_oracles_and_preserve_inputs() -> None:
    cutlass.cuda.initialize_cuda_context()
    indices = np.arange(_TILE_ITEMS, dtype=np.int32)
    keys_host = ((indices * 1_103_515_245 + 12_345) ^ (indices << 7)).astype(np.int32)
    values_host = np.arange(_TILE_ITEMS, dtype=np.int32)
    keys = torch.from_numpy(keys_host).cuda()
    values = torch.from_numpy(values_host).cuda()
    output = torch.zeros(
        5 * _TILE_ITEMS,
        dtype=torch.int32,
        device="cuda",
    )

    _run_topk(
        runtime.from_dlpack(keys),
        runtime.from_dlpack(values),
        runtime.from_dlpack(output),
    )
    torch.cuda.synchronize()
    segments = output.cpu().numpy().reshape(5, _TILE_ITEMS)

    k = 11
    valid_items = _TILE_ITEMS - 7
    valid_digits = _ordered_digits(keys_host[:valid_items])
    np.testing.assert_array_equal(
        np.sort(_ordered_digits(segments[0, :k])),
        np.sort(valid_digits)[-k:],
    )
    np.testing.assert_array_equal(
        np.sort(_ordered_digits(segments[1, :k])),
        np.sort(valid_digits)[:k],
    )
    assert set(zip(segments[1, :k], segments[2, :k])) <= set(
        zip(keys_host[:valid_items], values_host[:valid_items])
    )
    np.testing.assert_array_equal(segments[3], keys_host)
    np.testing.assert_array_equal(segments[4], values_host)
