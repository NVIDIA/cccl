// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_select.cuh>

#include <thrust/count.h>

#include <cuda/std/algorithm>

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
void select(nvbench::state& state, nvbench::type_list<T, InPlace>)
{
  using offset_t = int64_t;

  // Retrieve axis parameters
  const auto elements       = state.get_int64("Elements{io}");
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  auto generator = generate(elements, entropy);

  thrust::device_vector<T> in       = generator;
  thrust::device_vector<bool> flags = generator;
  thrust::device_vector<offset_t> num_selected(1);

  // TODO Extract into helper TU
  const auto selected_elements = thrust::count(flags.cbegin(), flags.cend(), true);
  thrust::device_vector<T> out(selected_elements, thrust::no_init);

  T* d_in                  = thrust::raw_pointer_cast(in.data());
  T* d_out                 = thrust::raw_pointer_cast(out.data());
  const bool* d_flags      = thrust::raw_pointer_cast(flags.data());
  offset_t* d_num_selected = thrust::raw_pointer_cast(num_selected.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_reads<bool>(elements);
  state.add_global_memory_writes<T>(selected_elements);
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
        cub::DeviceSelect::Flagged,
        "DeviceSelect::Flagged failed",
        d_in,
        d_flags,
        d_num_selected,
        static_cast<offset_t>(elements),
        env);
    }
    else
    {
      _CCCL_TRY_CUDA_API(
        cub::DeviceSelect::Flagged,
        "DeviceSelect::Flagged failed",
        static_cast<const T*>(d_in),
        d_flags,
        d_out,
        d_num_selected,
        static_cast<offset_t>(elements),
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

NVBENCH_BENCH_TYPES(select, NVBENCH_TYPE_AXES(fundamental_types, is_in_place))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "InPlace{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});
