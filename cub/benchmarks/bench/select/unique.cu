// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_select.cuh>

#include <cuda/std/algorithm>

#include <limits>

#include <look_back_helper.cuh>
#include <nvbench_helper.cuh>

// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5

#if !TUNE_BASE
template <typename InputT>
struct bench_policy_selector
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::select::select_if_policy
  {
    return {TUNE_THREADS_PER_BLOCK,
            TUNE_ITEMS_PER_THREAD,
            (TUNE_TRANSPOSE == 0 ? cub::BLOCK_LOAD_DIRECT : cub::BLOCK_LOAD_WARP_TRANSPOSE),
            (TUNE_LOAD == 0 ? cub::LOAD_DEFAULT : cub::LOAD_CA),
            cub::BLOCK_SCAN_WARP_SCANS,
            delay_constructor_policy};
  }
};
#endif // !TUNE_BASE

template <typename T, typename InPlace>
static void unique(nvbench::state& state, nvbench::type_list<T, InPlace>)
{
  using offset_t = int64_t;

  // Retrieve axis parameters
  const auto elements         = state.get_int64("Elements{io}");
  const auto max_segment_size = state.get_int64("MaxSegSize");

  thrust::device_vector<T> in = generate.uniform.key_segments(elements, /* min_segmented_size */ 1, max_segment_size);
  thrust::device_vector<T> out(elements, thrust::no_init);
  thrust::device_vector<offset_t> num_unique_out(1);

  T* d_in                = thrust::raw_pointer_cast(in.data());
  T* d_out               = thrust::raw_pointer_cast(out.data());
  offset_t* d_num_unique = thrust::raw_pointer_cast(num_unique_out.data());

  // Get number of unique elements for metrics
  cub::DeviceSelect::Unique(d_in, d_out, d_num_unique, static_cast<offset_t>(elements), ::cuda::std::equal_to<>{});
  cudaDeviceSynchronize();
  const offset_t num_unique = num_unique_out[0];

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(num_unique);
  state.add_global_memory_writes<offset_t>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(bench_policy_selector<T>{})
#endif // !TUNE_BASE
    );
    if constexpr (InPlace::value)
    {
      _CCCL_TRY_CUDA_API(
        cub::DeviceSelect::Unique,
        "select_unique failed",
        d_in,
        d_num_unique,
        static_cast<offset_t>(elements),
        ::cuda::std::equal_to<>{},
        env);
    }
    else
    {
      _CCCL_TRY_CUDA_API(
        cub::DeviceSelect::Unique,
        "select_unique failed",
        d_in,
        d_out,
        d_num_unique,
        static_cast<offset_t>(elements),
        ::cuda::std::equal_to<>{},
        env);
    }
  });
}

using ::cuda::std::false_type;
using ::cuda::std::true_type;
#ifdef TUNE_InPlace
using is_in_place = nvbench::type_list<TUNE_InPlace>; // expands to "false_type" or "true_type"
#else // !defined(TUNE_InPlace)
using is_in_place = nvbench::type_list<false_type, true_type>;
#endif // TUNE_InPlace

NVBENCH_BENCH_TYPES(unique, NVBENCH_TYPE_AXES(fundamental_types, is_in_place))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "InPlace{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8});
