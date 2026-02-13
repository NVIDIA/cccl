// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_run_length_encode.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <look_back_helper.cuh>
#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5

#if !TUNE_BASE
struct bench_encode_policy_selector
{
  [[nodiscard]] constexpr auto operator()(::cuda::arch_id /*arch*/) const
    -> cub::detail::reduce_by_key::reduce_by_key_policy
  {
    return {
      TUNE_THREADS,
      TUNE_ITEMS,
      TUNE_TRANSPOSE == 0 ? cub::BLOCK_LOAD_DIRECT : cub::BLOCK_LOAD_WARP_TRANSPOSE,
      TUNE_LOAD == 0 ? cub::LOAD_DEFAULT : cub::LOAD_CA,
      cub::BLOCK_SCAN_WARP_SCANS,
      delay_constructor_policy,
    };
  }
};
#endif // !TUNE_BASE

template <class T, class OffsetT, class RunLengthT>
static void rle(nvbench::state& state, nvbench::type_list<T, OffsetT, RunLengthT>)
{
  // Offset type large enough to represent any offset into the input sequence
  using offset_t = cub::detail::choose_signed_offset_t<OffsetT>;
  // Offset type large enough to represent the longest run in the sequence
  using run_length_t = RunLengthT;
  // Offset type large enough to represent the total number of runs in the sequence
  using num_runs_t = offset_t;

  using keys_input_it_t            = const T*;
  using unique_output_it_t         = T*;
  using run_length_input_it_t      = thrust::constant_iterator<run_length_t, offset_t>;
  using run_length_output_it_t     = run_length_t*;
  using num_runs_output_iterator_t = num_runs_t*;
  using equality_op_t              = ::cuda::std::equal_to<>;
  using reduction_op_t             = ::cuda::std::plus<>;
  using accum_t                    = run_length_t;

  const auto elements                    = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  constexpr std::size_t min_segment_size = 1;
  const std::size_t max_segment_size     = static_cast<std::size_t>(state.get_int64("MaxSegSize"));

  thrust::device_vector<num_runs_t> num_runs_out(1);
  thrust::device_vector<run_length_t> out_vals(elements);
  thrust::device_vector<T> out_keys(elements);
  thrust::device_vector<T> in_keys = generate.uniform.key_segments(elements, min_segment_size, max_segment_size);

  T* d_in_keys        = thrust::raw_pointer_cast(in_keys.data());
  T* d_out_keys       = thrust::raw_pointer_cast(out_keys.data());
  auto d_out_vals     = thrust::raw_pointer_cast(out_vals.data());
  auto d_num_runs_out = thrust::raw_pointer_cast(num_runs_out.data());
  run_length_input_it_t d_in_vals(run_length_t{1});

  std::uint8_t* d_temp_storage{};
  std::size_t temp_storage_bytes{};
  const offset_t num_items = static_cast<offset_t>(elements);

  auto dispatch_on_stream = [&](cudaStream_t stream) {
    return cub::detail::reduce_by_key::dispatch_streaming_reduce_by_key(
      d_temp_storage,
      temp_storage_bytes,
      d_in_keys,
      d_out_keys,
      d_in_vals,
      d_out_vals,
      d_num_runs_out,
      equality_op_t{},
      reduction_op_t{},
      num_items,
      stream
#if !TUNE_BASE
      ,
      bench_encode_policy_selector{}
#endif
    );
  };

  dispatch_on_stream(cudaStream_t{0});

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  dispatch_on_stream(cudaStream_t{0});
  cudaDeviceSynchronize();
  const num_runs_t num_runs = num_runs_out[0];

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(num_runs);
  state.add_global_memory_writes<run_length_t>(num_runs);
  state.add_global_memory_writes<num_runs_t>(1);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_on_stream(launch.get_stream());
  });
}

using run_length_types = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;

NVBENCH_BENCH_TYPES(rle, NVBENCH_TYPE_AXES(all_types, offset_types, run_length_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "RunLengthT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8});
