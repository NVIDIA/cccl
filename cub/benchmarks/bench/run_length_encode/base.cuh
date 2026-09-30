// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/device/device_run_length_encode.cuh>

#include <nvbench_helper.cuh>

#include "policy_selector.h"

//! @tparam RunLengthT Offset type large enough to represent the longest run in the sequence
template <class T, class OffsetT, class RunLengthT>
static void rle(nvbench::state& state, nvbench::type_list<T, OffsetT, RunLengthT>)
{
  // Offset type large enough to represent any offset into the input sequence and the total number of runs
  using offset_t = cub::detail::choose_signed_offset_t<OffsetT>;

  const auto elements                    = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  constexpr std::size_t min_segment_size = 1;
  const std::size_t max_segment_size     = static_cast<std::size_t>(state.get_int64("MaxSegSize"));

  thrust::device_vector<offset_t> num_runs_out(1);
  thrust::device_vector<RunLengthT> out_counts(elements);
  thrust::device_vector<T> out_keys(elements);
  thrust::device_vector<T> in_keys = generate.uniform.key_segments(elements, min_segment_size, max_segment_size);

  const T* d_in_keys       = thrust::raw_pointer_cast(in_keys.data());
  T* d_out_keys            = thrust::raw_pointer_cast(out_keys.data());
  RunLengthT* d_out_counts = thrust::raw_pointer_cast(out_counts.data());
  offset_t* d_num_runs_out = thrust::raw_pointer_cast(num_runs_out.data());

  // Run once to get num_runs for memory accounting
  (void) cub::DeviceRunLengthEncode::Encode(
    d_in_keys, d_out_keys, d_out_counts, d_num_runs_out, static_cast<OffsetT>(elements));
  cudaDeviceSynchronize();
  const offset_t num_runs = num_runs_out[0];

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(num_runs);
  state.add_global_memory_writes<RunLengthT>(num_runs);
  state.add_global_memory_writes<offset_t>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(bench_encode_policy_selector{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceRunLengthEncode::Encode,
      "Encode failed",
      d_in_keys,
      d_out_keys,
      d_out_counts,
      d_num_runs_out,
      static_cast<OffsetT>(elements),
      env);
  });
}

using run_length_types = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;

NVBENCH_BENCH_TYPES(rle, NVBENCH_TYPE_AXES(rle_key_types, offset_types, run_length_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "RunLengthT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8});
