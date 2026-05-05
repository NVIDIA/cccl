// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/device/device_segmented_reduce.cuh>

#include <cuda/std/type_traits>

#ifndef TUNE_BASE
#  define TUNE_ITEMS_PER_VEC_LOAD (1 << TUNE_ITEMS_PER_VEC_LOAD_POW2)
#endif

#if !TUNE_BASE
template <typename AccumT>
struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> ::cub::segmented_reduce_policy
  {
    constexpr int accum_size = int{sizeof(AccumT)};

    const auto [l_items, l_threads] =
      cub::detail::scale_mem_bound(TUNE_L_NOMINAL_4B_THREADS_PER_BLOCK, TUNE_L_NOMINAL_4B_ITEMS_PER_THREAD, accum_size);
    const auto s_items =
      cub::detail::scale_mem_bound(TUNE_L_NOMINAL_4B_THREADS_PER_BLOCK, TUNE_S_NOMINAL_4B_ITEMS_PER_THREAD, accum_size)
        .items_per_thread;
    const auto m_items =
      cub::detail::scale_mem_bound(TUNE_L_NOMINAL_4B_THREADS_PER_BLOCK, TUNE_M_NOMINAL_4B_ITEMS_PER_THREAD, accum_size)
        .items_per_thread;

    const auto rp = cub::agent_reduce_policy{
      l_threads, l_items, TUNE_ITEMS_PER_VEC_LOAD, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_LDG};
    return {rp,
            cub::warp_reduce_policy{rp.block_threads, TUNE_S_THREADS_PER_WARP, s_items, rp.vec_size, rp.load_modifier},
            cub::warp_reduce_policy{rp.block_threads, TUNE_M_THREADS_PER_WARP, m_items, rp.vec_size, rp.load_modifier}};
  }
};
#endif // !TUNE_BASE

template <typename T>
void fixed_size_segmented_reduce(nvbench::state& state, nvbench::type_list<T>)
{
  static constexpr bool is_argmin = std::is_same_v<op_t, cub::detail::arg_min>;

  using output_t = cuda::std::conditional_t<is_argmin, cuda::std::pair<int, T>, T>;
  using accum_t  = output_t;
  using init_t   = cuda::std::conditional_t<is_argmin, cub::detail::reduce::empty_problem_init_t<accum_t>, T>;

  // Retrieve axis parameters
  const size_t num_elements = static_cast<size_t>(state.get_int64("Elements{io}"));
  const size_t segment_size = static_cast<size_t>(state.get_int64("SegmentSize"));
  const size_t num_segments = std::max<std::size_t>(1, (num_elements / segment_size));
  const size_t elements     = num_segments * segment_size;

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<output_t> out(num_segments);

  const T* d_in   = thrust::raw_pointer_cast(in.data());
  output_t* d_out = thrust::raw_pointer_cast(out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<output_t>(num_segments);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(policy_selector<accum_t>{})
#endif
    );
    if constexpr (is_argmin)
    {
      _CCCL_TRY_CUDA_API(
        cub::DeviceSegmentedReduce::ArgMin,
        "Segmented ArgMin failed",
        d_in,
        d_out,
        static_cast<::cuda::std::int64_t>(num_segments),
        static_cast<int>(segment_size),
        env);
    }
    else
    {
      _CCCL_TRY_CUDA_API(
        cub::DeviceSegmentedReduce::Reduce,
        "Segmented reduce failed",
        d_in,
        d_out,
        static_cast<::cuda::std::int64_t>(num_segments),
        static_cast<int>(segment_size),
        op_t{},
        init_t{},
        env);
    }
  });
}

NVBENCH_BENCH_TYPES(fixed_size_segmented_reduce, NVBENCH_TYPE_AXES(value_types))
  .set_name("small")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("SegmentSize", nvbench::range(0, 4, 1));

NVBENCH_BENCH_TYPES(fixed_size_segmented_reduce, NVBENCH_TYPE_AXES(value_types))
  .set_name("medium")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("SegmentSize", nvbench::range(5, 8, 1));

NVBENCH_BENCH_TYPES(fixed_size_segmented_reduce, NVBENCH_TYPE_AXES(value_types))
  .set_name("large")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("SegmentSize", nvbench::range(9, 16, 1));
