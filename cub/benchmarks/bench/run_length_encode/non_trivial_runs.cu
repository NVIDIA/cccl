// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_run_length_encode.cuh>

#include <look_back_helper.cuh>
#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_TIME_SLICING ts 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5

#if !TUNE_BASE
struct bench_rle_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::rle::non_trivial_runs::rle_non_trivial_runs_policy
  {
    return {
      TUNE_THREADS,
      TUNE_ITEMS,
      TUNE_TRANSPOSE == 0 ? cub::BLOCK_LOAD_DIRECT : cub::BLOCK_LOAD_WARP_TRANSPOSE,
      TUNE_LOAD == 0 ? cub::LOAD_DEFAULT : cub::LOAD_CA,
      static_cast<bool>(TUNE_TIME_SLICING),
      cub::BLOCK_SCAN_WARP_SCANS,
      delay_constructor_policy,
    };
  }
};
#endif // !TUNE_BASE

template <class T, class OffsetT, class RunLengthT>
static void rle(nvbench::state& state, nvbench::type_list<T, OffsetT, RunLengthT>)
{
  using offset_t = cub::detail::choose_signed_offset_t<OffsetT>;

  const auto elements                    = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  constexpr std::size_t min_segment_size = 1;
  const std::size_t max_segment_size     = static_cast<std::size_t>(state.get_int64("MaxSegSize"));

  thrust::device_vector<offset_t> num_runs_out(1);
  thrust::device_vector<offset_t> out_offsets(elements);
  thrust::device_vector<RunLengthT> out_lengths(elements);
  thrust::device_vector<T> in_keys = generate.uniform.key_segments(elements, min_segment_size, max_segment_size);

  const T* d_in_keys        = thrust::raw_pointer_cast(in_keys.data());
  offset_t* d_out_offsets   = thrust::raw_pointer_cast(out_offsets.data());
  RunLengthT* d_out_lengths = thrust::raw_pointer_cast(out_lengths.data());
  offset_t* d_num_runs_out  = thrust::raw_pointer_cast(num_runs_out.data());

  {
    // Run once to get num_runs for memory accounting
    auto memory_env = cuda::std::execution::env{
#if !TUNE_BASE
      cuda::execution::tune(bench_rle_policy_selector{})
#endif // !TUNE_BASE
    };
    _CCCL_TRY_CUDA_API(
      cub::DeviceRunLengthEncode::NonTrivialRuns,
      "NonTrivialRuns failed",
      d_in_keys,
      d_out_offsets,
      d_out_lengths,
      d_num_runs_out,
      static_cast<OffsetT>(elements),
      memory_env);
    cudaDeviceSynchronize();
  }
  const OffsetT num_runs = num_runs_out[0];

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<OffsetT>(num_runs);
  state.add_global_memory_writes<OffsetT>(num_runs);
  state.add_global_memory_writes<OffsetT>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(bench_rle_policy_selector{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DeviceRunLengthEncode::NonTrivialRuns,
      "NonTrivialRuns failed",
      d_in_keys,
      d_out_offsets,
      d_out_lengths,
      d_num_runs_out,
      static_cast<OffsetT>(elements),
      env);
  });
}

using run_length_types = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;

NVBENCH_BENCH_TYPES(rle, NVBENCH_TYPE_AXES(all_types, offset_types, run_length_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "RunLengthT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8});
