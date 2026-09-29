// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! `cub::DeviceRunLengthDecode::Decode`: runs given by their values and lengths.

#include <cub/device/device_run_length_decode.cuh>

#include <cuda/std/cstdint>

#include <nvbench_helper.cuh>

#include "base.cuh"

template <typename T, typename OffsetT>
void decode(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  run_length_decode_bench_data<T, OffsetT> s(state);
  const thrust::device_vector<OffsetT> run_lengths = s.run_lengths();
  state.add_global_memory_reads<OffsetT>(s.num_runs);

  const T* d_run_values        = thrust::raw_pointer_cast(s.run_values.data());
  const OffsetT* d_run_lengths = thrust::raw_pointer_cast(run_lengths.data());
  T* d_out                     = thrust::raw_pointer_cast(s.out.data());

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const auto env = cub_bench_env(alloc, launch);
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceRunLengthDecode::Decode,
      "Decode failed",
      d_run_values,
      d_run_lengths,
      d_out,
      static_cast<cuda::std::int64_t>(s.num_runs),
      env);
  });
}

NVBENCH_BENCH_TYPES(decode, NVBENCH_TYPE_AXES(integral_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8, 16});
