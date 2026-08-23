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

from_dlpack = runtime.from_dlpack
Int32 = pytest.importorskip("cutlass.base_dsl.typing").Int32

pytestmark = [
    pytest.mark.backend_cutlass,
    pytest.mark.runtime,
    pytest.mark.gpu,
]

_BLOCK_THREADS = 64
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
_BEGIN_BIT = 4
_END_BIT = 12
_RANK_BITS = 4
_OUTPUT_SEGMENTS = 6


@pytest.fixture(scope="module", autouse=True)
def _isolated_provider_cache(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("cuda-coop-cutlass-radix-runtime")
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
def _radix_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    output: cute.Tensor,
):
    rank = cute.arch.thread_idx()[0]
    offset = rank * Int32(_ITEMS_PER_THREAD)

    common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=int)
    qualified_keys = cutlass_coop.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
    qualified_values = cutlass_coop.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
    common_keys[0] = keys_in[offset]
    common_keys[1] = keys_in[offset + Int32(1)]
    qualified_keys[0] = keys_in[offset]
    qualified_keys[1] = keys_in[offset + Int32(1)]
    qualified_values[0] = values_in[offset]
    qualified_values[1] = values_in[offset + Int32(1)]

    common_sorted = coop.radix_sort_keys(
        coop.this_block(),
        common_keys,
        begin_bit=Int32(_BEGIN_BIT),
        end_bit=Int32(_END_BIT),
    )
    fixed_storage = cutlass_coop.TempStorage(8192, alignment=16)
    pair_keys, pair_values = cutlass_coop.radix_sort_pairs(
        cutlass_coop.this_block(),
        qualified_keys,
        qualified_values,
        begin_bit=Int32(_BEGIN_BIT),
        end_bit=Int32(_END_BIT),
        descending=True,
        temp_storage=fixed_storage,
    )
    prefix = cutlass_coop.ThreadData(1, dtype=Int32)
    ranks = cutlass_coop.radix_rank(
        cutlass_coop.this_block(),
        qualified_keys,
        begin_bit=0,
        radix_bits=_RANK_BITS,
        exclusive_digit_prefix=prefix,
    )

    _store(output, 0, rank, common_keys)
    _store(output, 1, rank, common_sorted)
    _store(output, 2, rank, pair_keys)
    _store(output, 3, rank, pair_values)
    _store(output, 4, rank, ranks)
    prefix_offset = 5 * _TILE_ITEMS + rank * _ITEMS_PER_THREAD
    output[prefix_offset] = prefix[0]
    output[prefix_offset + Int32(1)] = prefix[0]


@cute.jit
def _run_radix(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    output: cute.Tensor,
):
    _radix_kernel(keys_in, values_in, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


def _rank_reference(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    digits = keys.view(np.uint32) & np.uint32((1 << _RANK_BITS) - 1)
    counts = np.bincount(digits, minlength=1 << _RANK_BITS)
    prefixes = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int32)
    seen = np.zeros(1 << _RANK_BITS, dtype=np.int32)
    ranks = np.empty(keys.size, dtype=np.int32)
    for index, digit in enumerate(digits):
        digit_index = int(digit)
        ranks[index] = prefixes[digit_index] + seen[digit_index]
        seen[digit_index] += 1
    return ranks, prefixes


def test_runtime_bits_pairs_rank_prefix_storage_and_fresh_results() -> None:
    cutlass.cuda.initialize_cuda_context()
    indices = np.arange(_TILE_ITEMS, dtype=np.int32)
    keys_host = ((indices * np.int32(53)) % np.int32(257)) - np.int32(128)
    values_host = indices * np.int32(17) + np.int32(3)
    keys = torch.from_numpy(keys_host.copy()).cuda()
    values = torch.from_numpy(values_host.copy()).cuda()
    output = torch.full(
        (_OUTPUT_SEGMENTS * _TILE_ITEMS,),
        -9999,
        dtype=torch.int32,
        device="cuda",
    )

    _run_radix(
        from_dlpack(keys),
        from_dlpack(values),
        from_dlpack(output),
    )
    torch.cuda.synchronize()
    observed = output.cpu().numpy().reshape(_OUTPUT_SEGMENTS, _TILE_ITEMS)

    np.testing.assert_array_equal(observed[0], keys_host)
    selected_digits = (keys_host.view(np.uint32) >> np.uint32(_BEGIN_BIT)) & np.uint32(
        (1 << (_END_BIT - _BEGIN_BIT)) - 1
    )
    ascending_order = np.argsort(selected_digits, kind="stable")
    descending_order = np.argsort(
        -selected_digits.astype(np.int64),
        kind="stable",
    )
    np.testing.assert_array_equal(observed[1], keys_host[ascending_order])
    np.testing.assert_array_equal(observed[2], keys_host[descending_order])
    np.testing.assert_array_equal(observed[3], values_host[descending_order])

    expected_ranks, expected_prefixes = _rank_reference(keys_host)
    np.testing.assert_array_equal(observed[4], expected_ranks)
    observed_prefixes = observed[5, ::_ITEMS_PER_THREAD]
    np.testing.assert_array_equal(
        observed_prefixes[: 1 << _RANK_BITS],
        expected_prefixes,
    )
    np.testing.assert_array_equal(keys.cpu().numpy(), keys_host)
    np.testing.assert_array_equal(values.cpu().numpy(), values_host)
