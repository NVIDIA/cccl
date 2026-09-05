# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime and code-generation probe for common block Histogram."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from examples.cutlass._runtime import require_runtime

_BLOCK_DIM = (8, 4, 2)
_BLOCK_THREADS = 64
_ITEMS_PER_THREAD = 3
_SAMPLE_COUNT = _BLOCK_THREADS * _ITEMS_PER_THREAD
_BINS = 97
_BINS_PER_THREAD = 2
_COUNTER_CAPACITY = _BLOCK_THREADS * _BINS_PER_THREAD
_HISTOGRAM_SEGMENTS = 4
PORTABLE_DTYPE_CASES = (
    ("uint8", "int32"),
    ("int32", "uint32"),
    ("uint32", "int64"),
    ("int64", "uint64"),
    ("uint64", "uint32"),
)
_PORTABLE_DTYPE_NAMES = {
    "uint8": "Uint8",
    "int32": "Int32",
    "uint32": "Uint32",
    "int64": "Int64",
    "uint64": "Uint64",
}
_DTYPE_BLOCK_THREADS = 32
_DTYPE_ITEMS_PER_THREAD = 2
_DTYPE_SAMPLE_COUNT = _DTYPE_BLOCK_THREADS * _DTYPE_ITEMS_PER_THREAD
_DTYPE_BINS = 31
_DTYPE_COUNTER_CAPACITY = _DTYPE_BLOCK_THREADS


def _store_samples(output, segment: int, rank, samples) -> None:
    offset = segment * _SAMPLE_COUNT + rank * _ITEMS_PER_THREAD
    output[offset] = samples[0]
    output[offset + 1] = samples[1]
    output[offset + 2] = samples[2]


def _store_counters(output, segment: int, rank, counters) -> None:
    offset = segment * _COUNTER_CAPACITY
    output[offset + rank] = counters[0]
    output[offset + rank + _BLOCK_THREADS] = counters[1]


def _store_dtype_items(output, segment: int, rank, items) -> None:
    offset = segment * _DTYPE_SAMPLE_COUNT + rank * _DTYPE_ITEMS_PER_THREAD
    output[offset] = items[0]
    output[offset + 1] = items[1]


def _store_dtype_counter(output, segment: int, rank, counters) -> None:
    output[segment * _DTYPE_COUNTER_CAPACITY + rank] = counters[0]


def _compiler_dtype(dtype_name: str, *, cutlass_typing) -> type:
    try:
        return getattr(cutlass_typing, _PORTABLE_DTYPE_NAMES[dtype_name])
    except KeyError as exc:
        supported = ", ".join(sorted(_PORTABLE_DTYPE_NAMES))
        raise ValueError(
            f"dtype name must be one of {{{supported}}}; got {dtype_name!r}"
        ) from exc


@functools.lru_cache(maxsize=len(PORTABLE_DTYPE_CASES))
def make_dtype_runner(
    sample_dtype_name: str,
    counter_dtype_name: str,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Build one common-versus-qualified Histogram dtype runner."""

    if (sample_dtype_name, counter_dtype_name) not in PORTABLE_DTYPE_CASES:
        supported = ", ".join(
            f"{sample}/{counter}" for sample, counter in PORTABLE_DTYPE_CASES
        )
        raise ValueError(
            "sample/counter dtype pair must be one of "
            f"{{{supported}}}; got {sample_dtype_name}/{counter_dtype_name}"
        )

    from cutlass.base_dsl import typing as cutlass_typing

    cutlass, cute, torch, from_dlpack, cutlass_coop = require_runtime()
    from cuda import coop as common_coop

    ordinary_sample_dtype = getattr(np, sample_dtype_name)
    ordinary_counter_dtype = getattr(np, counter_dtype_name)
    sample_type = _compiler_dtype(
        sample_dtype_name,
        cutlass_typing=cutlass_typing,
    )
    counter_type = _compiler_dtype(
        counter_dtype_name,
        cutlass_typing=cutlass_typing,
    )
    sample_torch_dtype = getattr(torch, sample_dtype_name)
    counter_torch_dtype = getattr(torch, counter_dtype_name)
    globals()["cute"] = cute
    globals()["common_coop"] = common_coop
    globals()["cutlass_coop"] = cutlass_coop

    @cute.kernel
    def _kernel(
        values: cute.Tensor,
        samples_output: cute.Tensor,
        histogram_output: cute.Tensor,
    ):
        rank, _, _ = cute.arch.thread_idx()
        common_samples = common_coop.ThreadData(
            _DTYPE_ITEMS_PER_THREAD,
            dtype=ordinary_sample_dtype,
        )
        qualified_samples = cutlass_coop.ThreadData(
            _DTYPE_ITEMS_PER_THREAD,
            dtype=sample_type,
        )
        input_offset = rank * _DTYPE_ITEMS_PER_THREAD
        common_samples[0] = values[input_offset]
        common_samples[1] = values[input_offset + 1]
        qualified_samples[0] = values[input_offset]
        qualified_samples[1] = values[input_offset + 1]

        common_group = common_coop.this_block()
        qualified_group = cutlass_coop.this_block()
        common_atomic = common_coop.histogram(
            common_group,
            common_samples,
            bins=_DTYPE_BINS,
            counter_dtype=ordinary_counter_dtype,
        )
        qualified_atomic = cutlass_coop.histogram(
            qualified_group,
            qualified_samples,
            bins=_DTYPE_BINS,
            counter_dtype=counter_type,
        )
        common_sort = common_coop.histogram(
            common_group,
            common_samples,
            bins=_DTYPE_BINS,
            counter_dtype=ordinary_counter_dtype,
            algorithm="sort",
        )
        qualified_sort = cutlass_coop.histogram(
            qualified_group,
            qualified_samples,
            bins=_DTYPE_BINS,
            counter_dtype=counter_type,
            algorithm="sort",
        )

        _store_dtype_items(samples_output, 0, rank, common_samples)
        _store_dtype_items(samples_output, 1, rank, qualified_samples)
        _store_dtype_counter(histogram_output, 0, rank, common_atomic)
        _store_dtype_counter(histogram_output, 1, rank, qualified_atomic)
        _store_dtype_counter(histogram_output, 2, rank, common_sort)
        _store_dtype_counter(histogram_output, 3, rank, qualified_sort)

    @cute.jit
    def _run(
        values: cute.Tensor,
        samples_output: cute.Tensor,
        histogram_output: cute.Tensor,
    ):
        _kernel(values, samples_output, histogram_output).launch(
            grid=(1, 1, 1),
            block=(_DTYPE_BLOCK_THREADS, 1, 1),
        )

    return (
        _run,
        torch,
        from_dlpack,
        cutlass,
        sample_torch_dtype,
        counter_torch_dtype,
    )


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, cutlass_coop, Int32 = require_runtime(
        include_int32=True
    )
    from cutlass.base_dsl.typing import Int64, Uint8

    from cuda import coop as common_coop

    globals()["cute"] = cute
    globals()["common_coop"] = common_coop
    globals()["cutlass_coop"] = cutlass_coop
    globals()["Int32"] = Int32
    globals()["Int64"] = Int64
    globals()["Uint8"] = Uint8

    @cute.kernel
    def _kernel(
        values: cute.Tensor,
        samples_output: cute.Tensor,
        histogram_output: cute.Tensor,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        rank = tidx + Int32(8) * (tidy + Int32(4) * tidz)
        common_samples = common_coop.ThreadData(_ITEMS_PER_THREAD, dtype=np.uint8)
        qualified_samples = cutlass_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=Uint8,
        )
        input_offset = rank * _ITEMS_PER_THREAD
        common_samples[0] = values[input_offset]
        common_samples[1] = values[input_offset + 1]
        common_samples[2] = values[input_offset + 2]
        qualified_samples[0] = values[input_offset]
        qualified_samples[1] = values[input_offset + 1]
        qualified_samples[2] = values[input_offset + 2]

        common_group = common_coop.this_block()
        qualified_group = cutlass_coop.this_block()
        common_atomic = common_coop.histogram(
            common_group,
            common_samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
            counter_dtype=int,
        )
        qualified_atomic = cutlass_coop.histogram(
            qualified_group,
            qualified_samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
        )
        common_sort = common_coop.histogram(
            common_group,
            common_samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
            counter_dtype=np.int64,
            algorithm="sort",
        )
        qualified_sort = cutlass_coop.histogram(
            qualified_group,
            qualified_samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
            counter_dtype=Int64,
            algorithm="sort",
        )

        # Observe both payloads after every transforming operation.
        _store_samples(samples_output, 0, rank, common_samples)
        _store_samples(samples_output, 1, rank, qualified_samples)
        _store_counters(histogram_output, 0, rank, common_atomic)
        _store_counters(histogram_output, 1, rank, qualified_atomic)
        _store_counters(histogram_output, 2, rank, common_sort)
        _store_counters(histogram_output, 3, rank, qualified_sort)

    @cute.jit
    def _run(
        values: cute.Tensor,
        samples_output: cute.Tensor,
        histogram_output: cute.Tensor,
    ):
        _kernel(values, samples_output, histogram_output).launch(
            grid=(1, 1, 1),
            block=_BLOCK_DIM,
        )

    return _run, torch, from_dlpack, cutlass


def run_example() -> dict[str, Any]:
    run, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    indices = torch.arange(_SAMPLE_COUNT, dtype=torch.int64)
    values_host = ((indices * 29 + indices // 5) % _BINS).to(torch.uint8)
    values = values_host.cuda()
    samples_output = torch.zeros(
        (2 * _SAMPLE_COUNT,),
        dtype=torch.uint8,
        device="cuda",
    )
    histogram_output = torch.zeros(
        (_HISTOGRAM_SEGMENTS * _COUNTER_CAPACITY,),
        dtype=torch.int64,
        device="cuda",
    )

    for _ in range(2):
        samples_output.zero_()
        histogram_output.fill_(-1)
        run(
            from_dlpack(values),
            from_dlpack(samples_output),
            from_dlpack(histogram_output),
        )
        torch.cuda.synchronize()

        observed_samples = samples_output.cpu().reshape(2, _SAMPLE_COUNT)
        torch.testing.assert_close(observed_samples[0], values_host, atol=0, rtol=0)
        torch.testing.assert_close(observed_samples[1], values_host, atol=0, rtol=0)

        counts = torch.bincount(
            values_host.to(torch.int64),
            minlength=_BINS,
        )
        expected = torch.zeros((_COUNTER_CAPACITY,), dtype=torch.int64)
        expected[:_BINS] = counts
        observed_histograms = histogram_output.cpu().reshape(
            _HISTOGRAM_SEGMENTS,
            _COUNTER_CAPACITY,
        )
        for observed in observed_histograms:
            torch.testing.assert_close(observed, expected, atol=0, rtol=0)

    return {
        "algorithms": ("atomic", "sort"),
        "bins": _BINS,
        "bins_per_thread": _BINS_PER_THREAD,
        "block_dim": _BLOCK_DIM,
        "input_preserved": True,
        "out_of_range_slots_zero": True,
        "repeat_launches": 2,
    }


def run_dtype_example(
    sample_dtype_name: str,
    counter_dtype_name: str,
) -> dict[str, Any]:
    """Run one portable Histogram dtype pair against a striped host oracle."""

    run, torch, from_dlpack, cutlass, sample_torch_dtype, counter_torch_dtype = (
        make_dtype_runner(sample_dtype_name, counter_dtype_name)
    )
    cutlass.cuda.initialize_cuda_context()

    sample_dtype = getattr(np, sample_dtype_name)
    counter_dtype = getattr(np, counter_dtype_name)
    indices = np.arange(_DTYPE_SAMPLE_COUNT, dtype=np.uint64)
    values_numpy = ((indices * 29 + indices // 5) % _DTYPE_BINS).astype(sample_dtype)
    values_host = torch.tensor(values_numpy.tolist(), dtype=sample_torch_dtype)
    values = values_host.cuda()
    samples_output = torch.zeros(
        (2 * _DTYPE_SAMPLE_COUNT,),
        dtype=sample_torch_dtype,
        device="cuda",
    )
    histogram_output = torch.ones(
        (4 * _DTYPE_COUNTER_CAPACITY,),
        dtype=counter_torch_dtype,
        device="cuda",
    )

    run(
        from_dlpack(values),
        from_dlpack(samples_output),
        from_dlpack(histogram_output),
    )
    torch.cuda.synchronize()

    np.testing.assert_array_equal(values.cpu().numpy(), values_numpy)
    observed_samples = samples_output.cpu().reshape(
        2,
        _DTYPE_SAMPLE_COUNT,
    )
    np.testing.assert_array_equal(observed_samples[0].numpy(), values_numpy)
    np.testing.assert_array_equal(observed_samples[1].numpy(), values_numpy)

    expected = np.zeros((_DTYPE_COUNTER_CAPACITY,), dtype=counter_dtype)
    expected[:_DTYPE_BINS] = np.bincount(
        values_numpy.astype(np.int64),
        minlength=_DTYPE_BINS,
    )
    observed_histograms = histogram_output.cpu().reshape(
        4,
        _DTYPE_COUNTER_CAPACITY,
    )
    for observed in observed_histograms:
        np.testing.assert_array_equal(observed.numpy(), expected)

    return {
        "algorithms": ("atomic", "sort"),
        "bins": _DTYPE_BINS,
        "common_qualified_exact": True,
        "counter_dtype": counter_dtype_name,
        "input_preserved": True,
        "out_of_range_slots_zero": True,
        "sample_dtype": sample_dtype_name,
    }
