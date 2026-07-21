// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <nvbench_helper.cuh>

// %//RANGE//% TUNE_RADIX_BITS bits 8:9:1
#define TUNE_RADIX_BITS 8

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32

#include "policy_selector.h"

template <typename T, typename OffsetT>
void radix_sort_keys(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using value_t = cub::NullType;
  if constexpr (!fits_in_default_shared_memory<T, value_t, OffsetT, cub::SortOrder::Ascending>())
  {
    return;
  }

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  thrust::device_vector<T> buffer_1 = generate(elements, entropy);
  thrust::device_vector<T> buffer_2(elements, thrust::no_init);

  const T* d_buffer_1 = thrust::raw_pointer_cast(buffer_1.data());
  T* d_buffer_2       = thrust::raw_pointer_cast(buffer_2.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(policy_selector<T, value_t, OffsetT>{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DeviceRadixSort::SortKeys,
      "SortKeys failed",
      d_buffer_1,
      d_buffer_2,
      static_cast<OffsetT>(elements),
      0,
      sizeof(T) * 8,
      env);
  });
}

NVBENCH_BENCH_TYPES(radix_sort_keys, NVBENCH_TYPE_AXES(fundamental_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.201"});
