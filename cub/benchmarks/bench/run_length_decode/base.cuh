// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

//! Shared setup for the `cub::DeviceRunLengthDecode` benchmarks. The runs have uniformly distributed lengths in
//! [1, MaxSegSize] and random values, and decode to Elements{io} items.

#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>

#include <cstddef>

#include <nvbench_helper.cuh>

template <typename T, typename OffsetT>
struct run_length_decode_bench_data
{
  thrust::device_vector<OffsetT> run_offsets{};
  thrust::device_vector<T> run_values{};
  thrust::device_vector<T> out{};
  std::size_t elements{};
  std::size_t num_runs{};

  explicit run_length_decode_bench_data(nvbench::state& state)
  {
    elements                               = static_cast<std::size_t>(state.get_int64("Elements{io}"));
    const std::size_t max_segment_size     = static_cast<std::size_t>(state.get_int64("MaxSegSize"));
    constexpr std::size_t min_segment_size = 1;

    run_offsets = generate.uniform.segment_offsets(elements, min_segment_size, max_segment_size);
    num_runs    = run_offsets.size() - 1;
    run_values  = generate(num_runs);
    out         = thrust::device_vector<T>(elements, thrust::no_init);

    state.add_element_count(elements);
    state.add_global_memory_reads<T>(num_runs);
    state.add_global_memory_writes<T>(elements);
  }

  // Run lengths computed from the run offsets
  thrust::device_vector<OffsetT> run_lengths() const
  {
    thrust::device_vector<OffsetT> lengths(run_offsets.size(), thrust::no_init);
    thrust::adjacent_difference(run_offsets.cbegin(), run_offsets.cend(), lengths.begin());
    lengths.erase(lengths.begin());
    return lengths;
  }
};
