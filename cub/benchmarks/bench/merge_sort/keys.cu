// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_merge_sort.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK_POW2 tpb 6:10:1

#ifndef TUNE_BASE
#  define TUNE_THREADS_PER_BLOCK (1 << TUNE_THREADS_PER_BLOCK_POW2)
#endif // TUNE_BASE

#if !TUNE_BASE
template <typename KeyT>
struct policy_selector
{
  _CCCL_API constexpr auto operator()(::cuda::arch_id /*arch*/) const -> cub::detail::merge_sort::merge_sort_policy
  {
    return cub::detail::merge_sort::merge_sort_policy{
      TUNE_THREADS_PER_BLOCK,
      cub::Nominal4BItemsToItems<KeyT>(TUNE_ITEMS_PER_THREAD),
      (TUNE_TRANSPOSE == 0 ? cub::BLOCK_LOAD_DIRECT : cub::BLOCK_LOAD_WARP_TRANSPOSE),
      (TUNE_LOAD == 0 ? cub::LOAD_DEFAULT : (TUNE_LOAD == 1 ? cub::LOAD_LDG : cub::LOAD_CA)),
      (TUNE_TRANSPOSE == 0 ? cub::BLOCK_STORE_DIRECT : cub::BLOCK_STORE_WARP_TRANSPOSE)};
  }
};
#endif // !TUNE_BASE

template <typename T, typename OffsetT>
void keys(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using key_t        = T;
  using compare_op_t = less_t;

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  thrust::device_vector<T> buffer_1 = generate(elements, entropy);
  thrust::device_vector<T> buffer_2(elements);

  key_t* d_buffer_1 = thrust::raw_pointer_cast(buffer_1.data());
  key_t* d_buffer_2 = thrust::raw_pointer_cast(buffer_2.data());

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
      cuda::execution::__tune(policy_selector<key_t>{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DeviceMergeSort::SortKeysCopy,
      "SortKeysCopy failed",
      d_buffer_1,
      d_buffer_2,
      static_cast<OffsetT>(elements),
      compare_op_t{},
      env);
  });
}

NVBENCH_BENCH_TYPES(keys, NVBENCH_TYPE_AXES(all_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});
