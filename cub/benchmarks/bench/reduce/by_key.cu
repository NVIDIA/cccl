// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_reduce.cuh>

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
struct bench_reduce_by_key_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
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

template <class KeyT, class ValueT, class OffsetT>
static void reduce_by_key(nvbench::state& state, nvbench::type_list<KeyT, ValueT, OffsetT>)
{
  using reduction_op_t = ::cuda::std::plus<>;

  const auto elements                    = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  constexpr std::size_t min_segment_size = 1;
  const std::size_t max_segment_size     = static_cast<std::size_t>(state.get_int64("MaxSegSize"));

  thrust::device_vector<OffsetT> num_runs_out(1);
  thrust::device_vector<ValueT> in_vals(elements);
  thrust::device_vector<ValueT> out_vals(elements);
  thrust::device_vector<KeyT> out_keys(elements);
  thrust::device_vector<KeyT> in_keys = generate.uniform.key_segments(elements, min_segment_size, max_segment_size);

  const KeyT* d_in_keys   = thrust::raw_pointer_cast(in_keys.data());
  KeyT* d_out_keys        = thrust::raw_pointer_cast(out_keys.data());
  const ValueT* d_in_vals = thrust::raw_pointer_cast(in_vals.data());
  ValueT* d_out_vals      = thrust::raw_pointer_cast(out_vals.data());
  OffsetT* d_num_runs_out = thrust::raw_pointer_cast(num_runs_out.data());

  caching_allocator_t alloc;

  // Run once to get the number of runs for reporting
  _CCCL_TRY_CUDA_API(
    cub::DeviceReduce::ReduceByKey,
    "ReduceByKey failed",
    d_in_keys,
    d_out_keys,
    d_in_vals,
    d_out_vals,
    d_num_runs_out,
    reduction_op_t{},
    static_cast<OffsetT>(elements),
    alloc);
  cudaDeviceSynchronize();
  const OffsetT num_runs = num_runs_out[0];

  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<ValueT>(num_runs);
  state.add_global_memory_writes<KeyT>(num_runs);
  state.add_global_memory_writes<OffsetT>(1);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(bench_reduce_by_key_policy_selector{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DeviceReduce::ReduceByKey,
      "ReduceByKey failed",
      d_in_keys,
      d_out_keys,
      d_in_vals,
      d_out_vals,
      d_num_runs_out,
      reduction_op_t{},
      static_cast<OffsetT>(elements),
      env);
  });
}

using some_offset_types = nvbench::type_list<nvbench::int32_t>;

#ifdef TUNE_KeyT
using key_types = nvbench::type_list<TUNE_KeyT>;
#else // !defined(TUNE_KeyT)
using key_types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t
#  if _CCCL_HAS_INT128()
                     ,
                     int128_t
#  endif
                     >;
#endif // TUNE_KeyT

#ifdef TUNE_ValueT
using value_types = nvbench::type_list<TUNE_ValueT>;
#else // !defined(TUNE_ValueT)
using value_types = all_types;
#endif // TUNE_ValueT

NVBENCH_BENCH_TYPES(reduce_by_key, NVBENCH_TYPE_AXES(key_types, value_types, some_offset_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8});
