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

_BLOCK_THREADS = 32
_HISTOGRAM_ITEMS_PER_THREAD = 2
_BINS = 17
_DECODED_ITEMS_PER_THREAD = 2
_DECODED_TILE_ITEMS = _BLOCK_THREADS * _DECODED_ITEMS_PER_THREAD
_WINDOW_OFFSET = 3
_OUT_OF_RANGE_OFFSET = 100


@pytest.fixture(scope="module", autouse=True)
def _isolated_provider_cache(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("cuda-coop-cutlass-histogram-run-length")
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
def _histogram_kernel(samples: cute.Tensor, output: cute.Tensor):
    rank = cute.arch.thread_idx()[0]
    offset = rank * Int32(_HISTOGRAM_ITEMS_PER_THREAD)
    common_samples = coop.ThreadData(_HISTOGRAM_ITEMS_PER_THREAD, dtype=int)
    qualified_samples = cutlass_coop.ThreadData(
        _HISTOGRAM_ITEMS_PER_THREAD,
        dtype=Int32,
    )
    common_samples[0] = samples[offset]
    common_samples[1] = samples[offset + Int32(1)]
    qualified_samples[0] = samples[offset]
    qualified_samples[1] = samples[offset + Int32(1)]

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
    output[rank] = common_counts[0]
    output[_BLOCK_THREADS + rank] = qualified_counts[0]


@cute.jit
def _run_histogram(samples: cute.Tensor, output: cute.Tensor):
    _histogram_kernel(samples, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


@cute.kernel
def _run_length_decode_kernel(
    values: cute.Tensor,
    lengths: cute.Tensor,
    output: cute.Tensor,
):
    rank = cute.arch.thread_idx()[0]
    common_values = coop.ThreadData(1, dtype=int)
    common_lengths = coop.ThreadData(1, dtype=int)
    qualified_values = cutlass_coop.ThreadData(1, dtype=Int32)
    qualified_lengths = cutlass_coop.ThreadData(1, dtype=Int32)
    common_values[0] = values[rank]
    common_lengths[0] = lengths[rank]
    qualified_values[0] = values[rank]
    qualified_lengths[0] = lengths[rank]

    common_decoded = coop.run_length_decode(
        coop.this_block(),
        common_values,
        common_lengths,
        decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
        decoded_window_offset=_WINDOW_OFFSET,
    )
    relative = cutlass_coop.ThreadData(
        _DECODED_ITEMS_PER_THREAD,
        dtype=Int32,
    )
    total = cutlass_coop.ThreadData(1, dtype=Int32)
    qualified_decoded = cutlass_coop.run_length_decode(
        cutlass_coop.this_block(),
        qualified_values,
        qualified_lengths,
        decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
        decoded_window_offset=_WINDOW_OFFSET,
        relative_offsets=relative,
        total_decoded_size=total,
    )
    beyond_total = cutlass_coop.run_length_decode(
        cutlass_coop.this_block(),
        qualified_values,
        qualified_lengths,
        decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
        decoded_window_offset=_OUT_OF_RANGE_OFFSET,
    )

    base = rank * _DECODED_ITEMS_PER_THREAD
    output[base] = common_decoded[0]
    output[base + 1] = common_decoded[1]
    output[_DECODED_TILE_ITEMS + base] = qualified_decoded[0]
    output[_DECODED_TILE_ITEMS + base + 1] = qualified_decoded[1]
    output[2 * _DECODED_TILE_ITEMS + base] = beyond_total[0]
    output[2 * _DECODED_TILE_ITEMS + base + 1] = beyond_total[1]
    output[3 * _DECODED_TILE_ITEMS + base] = relative[0]
    output[3 * _DECODED_TILE_ITEMS + base + 1] = relative[1]
    output[4 * _DECODED_TILE_ITEMS + rank] = total[0]


@cute.jit
def _run_run_length_decode(
    values: cute.Tensor,
    lengths: cute.Tensor,
    output: cute.Tensor,
):
    _run_length_decode_kernel(values, lengths, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


def test_common_atomic_and_qualified_sort_histograms_match_oracle() -> None:
    cutlass.cuda.initialize_cuda_context()
    samples_host = (
        torch.arange(
            _BLOCK_THREADS * _HISTOGRAM_ITEMS_PER_THREAD,
            dtype=torch.int32,
        )
        * 7
        + 3
    ) % _BINS
    samples = samples_host.cuda()
    output = torch.full(
        (2 * _BLOCK_THREADS,),
        -1,
        dtype=torch.int32,
        device="cuda",
    )

    _run_histogram(from_dlpack(samples), from_dlpack(output))
    torch.cuda.synchronize()

    expected = torch.zeros(_BLOCK_THREADS, dtype=torch.int32)
    expected[:_BINS] = torch.bincount(samples_host, minlength=_BINS)
    observed = output.cpu().reshape(2, _BLOCK_THREADS)
    torch.testing.assert_close(observed[0], expected, atol=0, rtol=0)
    torch.testing.assert_close(observed[1], expected, atol=0, rtol=0)
    torch.testing.assert_close(samples.cpu(), samples_host, atol=0, rtol=0)


def test_decode_side_outputs_and_out_of_range_window_match_oracle() -> None:
    cutlass.cuda.initialize_cuda_context()
    values_host = torch.arange(_BLOCK_THREADS, dtype=torch.int32) + 20
    lengths_host = torch.zeros(_BLOCK_THREADS, dtype=torch.int32)
    values_host[:4] = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
    lengths_host[:4] = torch.tensor([2, 1, 3, 4], dtype=torch.int32)
    values = values_host.cuda()
    lengths = lengths_host.cuda()
    output = torch.full(
        (4 * _DECODED_TILE_ITEMS + _BLOCK_THREADS,),
        -777,
        dtype=torch.int32,
        device="cuda",
    )

    _run_run_length_decode(
        from_dlpack(values),
        from_dlpack(lengths),
        from_dlpack(output),
    )
    torch.cuda.synchronize()

    stream = torch.tensor([7, 7, 8, 9, 9, 9, 10, 10, 10, 10])
    relative_stream = torch.tensor([0, 1, 0, 0, 1, 2, 0, 1, 2, 3])
    expected_decoded = torch.zeros(_DECODED_TILE_ITEMS, dtype=torch.int32)
    expected_relative = torch.full(
        (_DECODED_TILE_ITEMS,),
        -1,
        dtype=torch.int32,
    )
    remaining = stream[_WINDOW_OFFSET:]
    expected_decoded[: remaining.numel()] = remaining
    expected_relative[: remaining.numel()] = relative_stream[_WINDOW_OFFSET:]

    observed = output.cpu()
    torch.testing.assert_close(
        observed[:_DECODED_TILE_ITEMS],
        expected_decoded,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        observed[_DECODED_TILE_ITEMS : 2 * _DECODED_TILE_ITEMS],
        expected_decoded,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        observed[2 * _DECODED_TILE_ITEMS : 3 * _DECODED_TILE_ITEMS],
        torch.zeros_like(expected_decoded),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        observed[3 * _DECODED_TILE_ITEMS : 4 * _DECODED_TILE_ITEMS],
        expected_relative,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        observed[4 * _DECODED_TILE_ITEMS :],
        torch.full((_BLOCK_THREADS,), stream.numel(), dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(values.cpu(), values_host, atol=0, rtol=0)
    torch.testing.assert_close(lengths.cpu(), lengths_host, atol=0, rtol=0)
