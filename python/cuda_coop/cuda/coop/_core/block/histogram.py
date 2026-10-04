# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe fresh CUB histograms with striped per-thread counters.

The adapter copies samples because CUB's sort algorithm can reorder them. It
computes a fresh histogram in shared storage, then waits for all updates
before distributing the counters. That internal barrier makes the returned
counts ready; scratch reuse after the call is a separate frontend concern.

The wrapper uses threadIdx.x for bin ownership, so only one-dimensional blocks
are supported. All C++ source remains a compilation description here;
constructing a specialization does not compile or launch the operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._algorithm import Algorithm, TypeDefinition
from .._types import (
    INT32,
    Array,
    Dependency,
    TemplateParameter,
    TempStorageParameter,
)
from ._common import normalize_block_dim, normalize_positive_int

HISTOGRAM_SAMPLE_DTYPES = frozenset(
    {"uint8", "int32", "uint32", "int64", "uint64"}
)
HISTOGRAM_COUNTER_DTYPES = frozenset({"int32", "uint32", "int64", "uint64"})


def validate_histogram_dtype(dtype: Any, *, counter: bool = False) -> Any:
    """Accept a supported sample or counter dtype without importing a compiler.

    Inspect the dtype name and return its original representation, except that
    Python int maps to the core int32 type. Samples and counters have separate
    allowed sets: a uint8 sample is a bin index, but counters need at least 32
    bits to hold the bounded number of samples in a block tile.
    """

    if dtype is int:
        dtype = INT32
    name = getattr(dtype, "name", getattr(dtype, "__name__", dtype))
    allowed = HISTOGRAM_COUNTER_DTYPES if counter else HISTOGRAM_SAMPLE_DTYPES
    if name not in allowed:
        parameter = "counter_dtype" if counter else "samples"
        raise TypeError(
            f"histogram {parameter} dtype must be one of {sorted(allowed)}"
        )
    return dtype


def normalize_histogram_algorithm(algorithm: Any) -> str:
    """Accept the CUB counting strategy by its exact public name."""

    if not isinstance(algorithm, str) or algorithm not in {"atomic", "sort"}:
        raise ValueError("histogram algorithm must be 'atomic' or 'sort'")
    return algorithm


BLOCK_HISTOGRAM_TYPE = TypeDefinition(
    name="BlockHistogramCoop",
    code=r"""
namespace cub {
template <typename SampleT, int BlockDimX, int ItemsPerThread, int Bins,
          int BinsPerThread, typename CounterT, BlockHistogramAlgorithm Algorithm>
class BlockHistogramCoop
{
  using histogram_t = BlockHistogram<SampleT, BlockDimX, ItemsPerThread, Bins, Algorithm>;
  // CUDA's block atomicAdd supports unsigned 64-bit counters. A fresh
  // histogram contains at most BlockDimX * ItemsPerThread samples, which
  // the planner bounds by INT_MAX, so conversion to any exposed counter
  // type is exact.
  using internal_counter_t = ::cuda::std::conditional_t<sizeof(CounterT) == 8,
                                                      unsigned long long, unsigned int>;
public:
  struct TempStorage
  {
    typename histogram_t::TempStorage scratch;
    internal_counter_t counters[Bins];
  };
private:
  TempStorage& storage;
public:
  _CCCL_DEVICE_API _CCCL_FORCEINLINE explicit BlockHistogramCoop(TempStorage& temp_storage)
      : storage(temp_storage) {}

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Histogram(
      SampleT (&samples)[ItemsPerThread], CounterT (&counts)[BinsPerThread])
  {
    SampleT items[ItemsPerThread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      items[i] = samples[i];
    }
    histogram_t(storage.scratch).Histogram(items, storage.counters);
    __syncthreads();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < BinsPerThread; ++i)
    {
      const int bin = static_cast<int>(threadIdx.x) + i * BlockDimX;
      counts[i] = bin < Bins ? static_cast<CounterT>(storage.counters[bin]) : CounterT{0};
    }
  }
};
} // namespace cub
""".strip(),  # noqa: E501 - Embedded C++ source.
)


@dataclass(frozen=True)
class BlockHistogramSpecialization:
    """Pair a CUB compilation description with input and output geometry.

    Input items_per_thread and output bins_per_thread are independent, so the
    result need not have the samples' shape. Current planners use only
    ``specialization``; the shared result contract carries the counter extent.
    """

    specialization: Algorithm
    block_dim: tuple[int, int, int]
    items_per_thread: int
    bins: int
    bins_per_thread: int


def make_block_histogram_specialization(
    *,
    sample_dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: Any = INT32,
    algorithm: str = "atomic",
) -> BlockHistogramSpecialization:
    """Validate histogram geometry and describe the specialized CUB adapter.

    Require a one-dimensional block and enough output slots to cover every
    bin. The samples per block (block size times items_per_thread) and output
    slots per block (block size times bins_per_thread) must fit a signed
    32-bit integer. Also validate both dtypes and the algorithm name. The
    input size bounds fresh bin counts, so the wrapper can safely convert
    its unsigned internal counters to any exposed counter dtype.

    The Algorithm takes sample and counter arrays with independent dtypes and
    extents. Its shared storage contains both CUB scratch and intermediate bin
    counters. The wrapper copies bins into striped per-thread output and
    writes zero into output slots beyond the bin count.
    """

    block_dim = normalize_block_dim(block_dim)
    if block_dim[1:] != (1, 1):
        raise ValueError("histogram supports only one-dimensional blocks")
    items_per_thread = normalize_positive_int(
        "items_per_thread", items_per_thread
    )
    bins = normalize_positive_int("bins", bins)
    bins_per_thread = normalize_positive_int("bins_per_thread", bins_per_thread)
    if bins > block_dim[0] * bins_per_thread:
        raise ValueError(
            "histogram bins_per_thread must provide capacity for every bin"
        )
    if (
        max(
            bins,
            block_dim[0] * bins_per_thread,
            block_dim[0] * items_per_thread,
        )
        > (1 << 31) - 1
    ):
        raise ValueError(
            "histogram tile and output sizes must fit a signed 32-bit integer"
        )
    sample_dtype = validate_histogram_dtype(sample_dtype)
    counter_dtype = validate_histogram_dtype(counter_dtype, counter=True)
    algorithm = normalize_histogram_algorithm(algorithm)
    specialization = Algorithm(
        struct_name="BlockHistogramCoop",
        method_name="Histogram",
        c_name="block_histogram",
        includes=(
            "cub/block/block_histogram.cuh",
            "cuda/std/__type_traits/conditional.h",
        ),
        template_parameters=tuple(
            TemplateParameter(name)
            for name in (
                "SampleT",
                "BLOCK_DIM_X",
                "ITEMS_PER_THREAD",
                "BINS",
                "BINS_PER_THREAD",
                "CounterT",
                "ALGORITHM",
            )
        ),
        parameters=(
            (
                TempStorageParameter(),
                Array(
                    Dependency("SampleT"),
                    Dependency("ITEMS_PER_THREAD"),
                    name="samples",
                ),
                Array(
                    Dependency("CounterT"),
                    Dependency("BINS_PER_THREAD"),
                    name="counts",
                    is_output=True,
                    is_return=False,
                ),
            ),
        ),
        type_definitions=(BLOCK_HISTOGRAM_TYPE,),
        template_arguments={
            "SampleT": sample_dtype,
            "BLOCK_DIM_X": block_dim[0],
            "ITEMS_PER_THREAD": items_per_thread,
            "BINS": bins,
            "BINS_PER_THREAD": bins_per_thread,
            "CounterT": counter_dtype,
            "ALGORITHM": "::cub::BLOCK_HISTO_ATOMIC"
            if algorithm == "atomic"
            else "::cub::BLOCK_HISTO_SORT",
        },
        metadata={
            "scope": "block",
            "primitive": "histogram",
            "algorithm": algorithm,
        },
    )
    return BlockHistogramSpecialization(
        specialization, block_dim, items_per_thread, bins, bins_per_thread
    )
